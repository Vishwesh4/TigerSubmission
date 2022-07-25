from sklearn.utils import shuffle
import torch, os
import torch._utils
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import json
import argparse
from pathlib import Path
import time
import torchvision
import torch.nn as nn
import wandb
from utils.dataloader import Tumourbed_dataset_train
from utils.parallel import DataParallelModel, DataParallelCriterion, gather
import numpy as np
import torchmetrics
import random
import pandas as pd
import torchvision.transforms as transforms

################################################################ Load arguments #################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-p", help="hyperparameter location",required=True)
parser.add_argument("-n", help="project name",required=True)
#For compute canada
parser.add_argument("-l",help="Data location modfier",default=None)
parser.add_argument("-m",help="(True/False) To use dataparallel or not",default="True")
parser.add_argument("-r", help="resume location",default=None)
parser.add_argument("-s",help="(True/False)save checkpoints?",default="False")
parser.add_argument("-w",help="(True/False)save wandb?",default="True")
parser.add_argument("-t",help="Path for loading weights for transfer learning",default=None)

args = parser.parse_args()

hyp = args.p
loc_mod = args.l
name = args.n
multi_gpu = eval(args.m)
resume_location = args.r
is_save = eval(args.s)
is_wandb = eval(args.w)
transfer_load = args.t

with open(hyp,"r") as f: 
	data_hyp=json.load(f) 
print(f"Experiment Name: {name}\nHyperparameters:{data_hyp}")
print("CUDA is available:{}".format(torch.cuda.is_available()))

if loc_mod is None:
    dataset_path = Path(data_hyp["DATASET_PATH"])
else:
    dataset_path = Path(loc_mod)/Path(data_hyp["DATASET_PATH"])

if is_save:
    PARENT_NAME = Path(data_hyp["SAVE_DIR"])
    FOLDER_NAME = PARENT_NAME / Path("Results")
    MODEL_SAVE = FOLDER_NAME / Path(name) / Path("saved_models")

    if not FOLDER_NAME.is_dir():
        os.mkdir(FOLDER_NAME)
        os.mkdir(MODEL_SAVE.parent)
        os.mkdir(MODEL_SAVE)
    elif not MODEL_SAVE.parent.is_dir():
        os.mkdir(MODEL_SAVE.parent)
        os.mkdir(MODEL_SAVE)
    else:
        pass

random_seed = 2022
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
#########################################################################################################################################################

def multifocal_loss(test_input,test_target,gamma=2):
    ce_loss = torch.nn.functional.cross_entropy(test_input, test_target, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()

def save_checkpoint(epoch,model,optimizer,scheduler,metric):
    """
    Saves checkpoint for the model, optimizer, scheduler. Additionally saves the best metric score and epoch value
    """
    print("Saving Checkpoint ...")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metric': metric
        }, MODEL_SAVE/Path("Checkpoint_{}.pt".format(time.strftime("%d%b%H_%M_%S",time.gmtime())))
    )
    torch.cuda.empty_cache()

def load_checkpoint(path,model,optimizer,scheduler):
    """
    Loads the saved checkpoint
    """
    checkpoint = torch.load(path, map_location=f'cuda:{DEVICE_LIST[0]}')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    metric = checkpoint['metric']
    del checkpoint
    torch.cuda.empty_cache()

    return model,optimizer,scheduler,epoch,metric

def load_model_weights(model, model_path):
    """
    Loads model weight
    """
    state = torch.load(model_path, map_location=f'cuda:{DEVICE_LIST[0]}')
    state_dict = state['model']
    for key in list(state_dict.keys()):
        state_dict[key.replace('model.', '').replace('resnet.', '').replace('module.', '')] = state_dict.pop(key)
    model_dict = model.state_dict()
    weights = {k: v for k, v in state_dict.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)

    return model

def train(model,trainloader, optimizer, LossFun,MetricFun):
    running_loss = 0.0
    model.train()
    for data in tqdm(trainloader):
        inputs, labels = data
        labels = labels.type(torch.LongTensor)
        #Convert to GPU
        inputs,labels = inputs.to(device), labels.to(device)

        # initialize gradients to zero
        optimizer.zero_grad() 
        # forward pass
        outputs = model(inputs)
        if torch.cuda.device_count() > 1 and multi_gpu:
            outputs = gather(outputs,device)
        #compute Loss with respect to target
        loss = LossFun(outputs, labels)
        # _, predicted = torch.max(outputs.data, 1)
        #Metric Calculation
        # total_dice+=MetricFun(outputs, labels)
        metrics_calc = MetricFun(outputs,labels.type(torch.int16))
        # back propagate
        loss.backward()
        # do SGD step i.e., update parameters
        optimizer.step()
        # by default loss is averaged over all elements of batch
        running_loss += loss.data

        if is_wandb:
            wandb.log({
                "Epoch Train Loss":loss.data,
            })        

    running_loss = running_loss.cpu().numpy()
    metrics_calc = MetricFun.compute()
    # print(metrics_calc)
    MetricFun.reset()
    F1score = metrics_calc["train_F1Score"].cpu().numpy().item()
    Accuracy = metrics_calc["train_Accuracy"].cpu().numpy().item()
    if is_wandb:
        wandb.log({
            "Epoch Train Total Accuracy": Accuracy,
            "Epoch F1 Score": F1score
        })

    return running_loss

def test(model,testloader,LossFun,epoch,MetricFun):
    running_loss = 0.0
    total_dice = 0
    # evaluation mode which takes care of architectural disablings
    model.eval()
    with torch.no_grad():
        for data in tqdm(testloader):
            inputs, labels = data
            labels = labels.type(torch.LongTensor)
            #Convert to GPU
            inputs,labels = inputs.to(device),labels.to(device)
            outputs = model(inputs)
            if torch.cuda.device_count() > 1 and multi_gpu:
                outputs = gather(outputs,device)
            #due to multi gpu, output is (Nx1) and label is (N,)
            loss = LossFun(outputs, labels)
            # _, predicted = torch.max(outputs.data, 1)
            #Metric Calculation
            # total_dice+=MetricFun(outputs, labels)
            metrics_calc = MetricFun(outputs,labels.type(torch.int16))
            running_loss += loss.data
    
    metrics_calc = MetricFun.compute()
    # print(metrics_calc)
    running_loss = running_loss.cpu().numpy()
    MetricFun.reset()
    F1score = metrics_calc["test_F1Score"].cpu().numpy().item()
    Accuracy = metrics_calc["test_Accuracy"].cpu().numpy().item()
    print(f"Total Accuracy score on test data : {Accuracy}")
    if is_wandb and epoch%5==0:
        log_wandb_table_stats(inputs,outputs,labels,epoch,metrics_calc)
    if is_wandb:
        wandb.log({
            "Test Loss":running_loss /  len(testloader),
            "Test F1score": F1score,
            "Test Accuracy":  Accuracy
        })

    return F1score , running_loss /  len(testloader)

def log_wandb_table_stats(Input,Output,Truth,epoch,metrics_calc):
    # W&B: Create a Table to store predictions for each test step
    columns=["id", "image","Calculated Class","Real Class","True Negative","False Positive","False Negative","True Positive"]
    test_table = wandb.Table(columns=columns)
    # Y_pred_target=torch.argmax(Output,dim=1)
    # Y_true_target=torch.argmax(Truth[0][:,:-1,:,:],dim=1).cpu().numpy()
    # metric_calc = MetricFun(Output.ravel(), Truth[1])
    # r2score = metric_calc["test_R2Score"].cpu().numpy()
    # mse = metric_calc["test_MeanSquaredError"].cpu().numpy()
    # metrics_calc = MetricFun(Output,Truth.type(torch.int16))
    Confusion = metrics_calc["test_ConfusionMatrix"].cpu().numpy()
    _, predicted = torch.max(Output.data, 1)
    for i in range(16):
        idx = f"{epoch}_{i}"
        image = wandb.Image(Input[i].permute(1,2,0).cpu().numpy())

        # mask = wandb.Image(image, masks={
        #     "Tissue mask": {"mask_data" : Y_true_target[i], "class_labels" : CLASSES},
        #     "Cell mask": {"mask_data" : Truth[0][i,-1,:,:].cpu().numpy(), "class_labels" : {0:"None",1:"Lymphocytes and plasma cells"}}
        # })
        # metric_calc = MetricFun(Output[i][0], Truth[1][i])
        test_table.add_data(idx, image,predicted[i],Truth[i],Confusion[0,0],Confusion[0,1],Confusion[1,0],Confusion[1,1])
    wandb.log({"table_key": test_table})
    # MetricFun.reset()

if __name__=="__main__":
    ###################################
    #          Model Setup            #
    ###################################
    NUM_CLASSES = 2
    DEVICE_LIST = data_hyp["DEVICE_LIST"]
    NUM_GPUS = len(DEVICE_LIST)
    # CLASSES = {0:'non-enhancing tumor core',1:'peritumoral edema',2:'GD-enhancing tumor',3:'background'}
    CLASSES = {0:'Non Relevant',1:'Relevant'}
    device = torch.device(f"cuda:{DEVICE_LIST[0]}" if torch.cuda.is_available() else "cpu")

    if (transfer_load is not None) and (resume_location is None):
        print("Loading pretraining weights from previous experiments...")
        model = torchvision.models.__dict__[data_hyp["ENCODER_NAME"]](pretrained=False)
        model = load_model_weights(model, transfer_load)
    else:
        model = torchvision.models.__dict__[data_hyp["ENCODER_NAME"]](pretrained=True)

    model.fc = nn.Sequential(
                torch.nn.Linear(model.fc.in_features, 300),
                torch.nn.Linear(300, NUM_CLASSES)
                )

    #Multiple GPUs
    if torch.cuda.device_count() > 1  and multi_gpu:
        print("Using {} GPUs".format(NUM_GPUS))
        model = DataParallelModel(model, device_ids=DEVICE_LIST)
    
    model = model.to(device)

    ###################################
    #        Data Augmentations       #
    ###################################   
    
    transform_train =  torchvision.transforms.Compose([    
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((256,256)),
        torchvision.transforms.RandAugment(),
        torchvision.transforms.ColorJitter(brightness=0.5,hue=0.3,saturation=0.3,contrast=0.3),
        torchvision.transforms.ToTensor()])
    # transform_train =  albumentations.Compose([
    #     ToTensorV2()]) 
    transform_test =  torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((256,256)),
        torchvision.transforms.ToTensor()])
    ###################################
    #          Data Loaders           #
    ###################################  


    train_dataset = Tumourbed_dataset_train(train_split=0.87,mode="training",data_path=dataset_path,transform=transform_train,seed=random_seed)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=data_hyp["TRAIN_BATCH_SIZE"],shuffle=True, num_workers=8,pin_memory=True)
    #Loading the images for test set
    # testset = Tumourbed_dataset_train(transform=transform_test)
    test_dataset = Tumourbed_dataset_train(train_split=0.87,mode="testing",data_path=dataset_path,transform=transform_test,seed=random_seed)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=data_hyp["TEST_BATCH_SIZE"],shuffle=True, num_workers=8,pin_memory=True)
    print(f"Total Training Images: {len(train_dataset)}\nTotal Testing Images: {len(test_dataset)}")
    ###########################################
    #  Loss function, optimizers and metrics  #
    ########################################### 
    
    #Loss function
    # LossFun = MultiFocalLoss(alpha=data_hyp['FOCAL_IMP'],gamma=data_hyp['FOCAL_GAMMA'])
    # LossFun = torch.nn.MSELoss()
    # LossFun = torch.nn.CrossEntropyLoss()
    LossFun = multifocal_loss
    # LossFun = nn.CrossEntropyLoss().cuda(device)
    # LossFun = LossFun.to(device)
    #optimizer
    optimizer = optim.Adam(model.parameters(),lr=data_hyp['LEARNINGRATE'], weight_decay = data_hyp['LAMBDA'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=data_hyp["PATIENCE"])
    #Metrics
    MetricFun = torchmetrics.MetricCollection([torchmetrics.Accuracy(dist_sync_on_step=multi_gpu),
                                            torchmetrics.F1Score(dist_sync_on_step=multi_gpu),
                                            torchmetrics.ConfusionMatrix(num_classes=NUM_CLASSES,dist_sync_on_step=multi_gpu)])
    #                                         torchmetrics.F1Score(num_classes=NUM_CLASSES,average=None,dist_sync_on_step=multi_gpu)])
    # metrics = torchmetrics.MetricCollection([DiceScore(labels=[0,1,2],dist_sync_on_step=multi_gpu)])
    
    TrainMetricDict = MetricFun.clone(prefix='train_').to(device)
    TestMetricDict = MetricFun.clone(prefix='test_').to(device)
    # DiceScore = torchmetrics.functional.dice_score
    # dicescore = DiceScore(labels=[0,1,2])
    # dicescore = compute_meandice_multilabel
    best_metric = -10000
    ###################################
    #           Training              #
    ###################################    
    if is_wandb:    
    #For visualizations
        wandb.init(project="Tumourbed detection",config=data_hyp)
        wandb.run.name = name
        wandb.run.save()
        # wandb.watch(model, log='all') 

    if resume_location is not None:
        model,optimizer,scheduler,epoch_start,best_metric = load_checkpoint(resume_location,model,optimizer,scheduler)
        # best_metric = best_metric.cpu().numpy()
        print("Resuming from saved checkpoint...")
    else:
        epoch_start = 0
    
    for epoch in range(epoch_start,data_hyp['EPOCHS']):
        print("EPOCH: {}".format(epoch))
        train(model, train_loader ,optimizer, LossFun,TrainMetricDict)
        metric,test_loss = test(model,test_loader,LossFun,epoch,TestMetricDict)
        ### Saving model checkpoints
        if is_save and (epoch+1) % 15 == 0 and metric>best_metric:
            save_checkpoint(epoch,model,optimizer,scheduler,metric)
            best_metric = metric
        scheduler.step(test_loss)
        if is_wandb:
            wandb.log({"Learning rate":optimizer.param_groups[0]["lr"]})
    print("Training done...")