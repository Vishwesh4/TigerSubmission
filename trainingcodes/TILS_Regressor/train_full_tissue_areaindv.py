import torch, os
import torch._utils
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import json
import argparse
import wandb
from pathlib import Path
import time
import albumentations
from albumentations.pytorch import ToTensorV2
from wholeslidedata.iterators import create_batch_iterator
from utils.model import Resnet_bihead_v2
from utils.dataloader import process_input_bihead
from utils.parallel import DataParallelModel, DataParallelCriterion,gather
from utils.edit_yaml import data_location_modifier, change_seed
# from utils.customHSV import HSV
import numpy as np
import torchmetrics
import random
import warnings
import gc

################################################################ Load arguments #################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-p", help="hyperparameter location",required=True)
parser.add_argument("-n", help="project name",required=True)
parser.add_argument("-m",help="(True/False) To use multiple-gpu or not",default="True")
parser.add_argument("-l",help="Data location modfier",default=None)
parser.add_argument("-s",help="(True/False)save checkpoints?",default="False")
parser.add_argument("-w",help="(True/False)save wandb?",default="True")
parser.add_argument("-t",help="Path for loading weights for transfer learning",default=None)
parser.add_argument("-r", help="resume location",default=None)
args = parser.parse_args()

hyp = args.p
name = args.n
multi_gpu = eval(args.m)
loc_mod = args.l
resume_location = args.r
is_save = eval(args.s)
is_wandb = eval(args.w)
transfer_load = args.t

#########################################################################################################################################################
with open(hyp,"r") as f: 
	data_hyp=json.load(f) 
print(f"Experiment Name: {name}\nHyperparameters:{data_hyp}")
print("CUDA is available:{}".format(torch.cuda.is_available()))


#For visualizations
if is_wandb:
    #For visualizations
    wandb.init(project="Cellularity and Tissue area",config=data_hyp,settings=wandb.Settings(start_method="fork"))
    wandb.config.update(args)
    wandb.run.name = name
    wandb.run.save()

if loc_mod is not None:
    dataset_path = Path(loc_mod)
    data_location_modifier(loc_mod,data_hyp['DATALOADER_CONFIG'])

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
def save_checkpoint(epoch,model,optimizer,scheduler,metric):
    print("Saving Checkpoint ...")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metric': metric
        }, MODEL_SAVE/Path("Checkpoint_{}_{:.2f}.pt".format(time.strftime("%d%b%H_%M_%S",time.gmtime()),metric.item()))
    )
    torch.cuda.empty_cache()

def load_checkpoint(path,model,optimizer,scheduler):
    checkpoint = torch.load(path, map_location=f'cuda:{DEVICE_LIST[0]}')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    metric = checkpoint['metric']
    del checkpoint
    torch.cuda.empty_cache()

    return model,optimizer,scheduler,epoch,metric

def load_model_weights(model, model_path,device):
    """
    Loads model weight
    """
    state = torch.load(model_path, map_location=device)
    possible_keys = ["model","model_state_dict","state_dict"]
    for p_keys in possible_keys:
        if p_keys in state:
            state_dict = state[p_keys]
            break
    for key in list(state_dict.keys()):
        state_dict[key.replace('resnet.', '').replace('module.', '')] = state_dict.pop(key)
        #For ozan model i think
        # state_dict[key.replace('model.', '').replace('resnet.', '').replace('module.', '')] = state_dict.pop(key)
    model_dict = model.state_dict()   
    weights = {k: v for k, v in state_dict.items() if k in model_dict}
    # weights = {k: v for k, v in state_dict.items() if (k in model_dict) & ("tissue" not in k)}
    if len(weights.keys())!=len(model_dict.keys()):
        warnings.warn("Warning... Some Weights could not be loaded")
        print("Warning... Some Weights could not be loaded")
    if weights == {}:
        warnings.warn("Warning... No weight could be loaded..")
        print("Warning... Some Weights could not be loaded")
    model_dict.update(weights)
    model.load_state_dict(model_dict)

    return model

def train(model, optimizer, LossFun,MetricFun):
    running_loss = 0.0
    # correct = 0
    # total = 0
    total_dice = 0
    tiger_dice = 0
    model.train()
    #To ensure each epoch model sees new set of images
    change_seed(config_file)
    with create_batch_iterator(mode="training", user_config=config_file, cpus=8) as trainloader:
        for _ in tqdm(range(data_hyp['TRAINREPEATS'])):
            data = next(trainloader)
            inputs, labels = process_input_bihead(data[0],data[1],transform_train,labels=[0,1,5,6],labels_dens=[1,2])
            #Convert to GPU
            inputs = inputs.to(device)
            labels[1] = labels[1].to(device)

            # initialize gradients to zero
            optimizer.zero_grad() 
            # forward pass
            outputs = model(inputs,"tissue")
            if torch.cuda.device_count() > 1 and multi_gpu:
                outputs = gather(outputs,device)
            #compute Loss with respect to target
            loss = LossFun(outputs.ravel(), labels[1])
            metrics_calc = MetricFun(outputs.ravel(),labels[1])
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
    R2score = np.array([metrics_calc["train_R2Score"].cpu().numpy()])
    if is_wandb:
        wandb.log({
            "Epoch Train Total R2 Score": R2score[0]
        })
    gc.collect()
    return running_loss

def test(model,LossFun,epoch,MetricFun):
    running_loss = 0.0
    total_dice = 0
    tiger_dice = 0
    # evaluation mode which takes care of architectural disablings
    model.eval()
    change_seed(config_file)
    with torch.no_grad():
        with create_batch_iterator(mode="inference", user_config=config_file, cpus=8) as testloader:  
        # with create_batch_iterator(mode="training", user_config=config_file, cpus=8) as testloader:  
            for _ in tqdm(range(data_hyp['TESTREPEATS'])):
                data = next(testloader)
                inputs, labels = process_input_bihead(data[0],data[1],transform_test,labels=[0,1,5,6],labels_dens=[1,2])
                #Convert to GPU
                inputs = inputs.to(device)
                labels[1] = labels[1].to(device)
                # with torch.no_grad():
                #     images = Variable(images)
                #     labels = Variable(labels)
                # if torch.cuda.is_available():
                #     images, labels = images.cuda(), labels.cuda()

                #Convert to GPU
                inputs,labels[1] = inputs.to(device),labels[1].to(device)
                outputs = model(inputs,"tissue")
                if torch.cuda.device_count() > 1 and multi_gpu:
                    outputs = gather(outputs,device)
                loss = LossFun(outputs.ravel(), labels[1])
                #Metric Calculation
                # total_dice+=MetricFun(outputs, labels)
                metrics_calc = MetricFun(outputs.ravel(),labels[1])
                running_loss += loss.data

    metrics_calc = MetricFun.compute()
    running_loss = running_loss.cpu().numpy()

    MetricFun.reset()
    R2score = np.array([metrics_calc["test_R2Score"].cpu().numpy()])
    # print(metrics_calc)
    # print(f"Total R2 score on test data : {metrics_calc['test_R2Score']}")
    # R2score = np.array([metrics_calc["test_R2Score"].cpu().numpy()])
    if is_wandb and epoch%5==0:
        log_wandb_table_stats(inputs,outputs,labels,epoch,MetricFun)
    if is_wandb:
        wandb.log({
            "Test Loss":running_loss /  data_hyp['TESTREPEATS'],
            "Test R2 Score":R2score[0]
        })

    return R2score[0] , running_loss /  data_hyp['TESTREPEATS']

def log_wandb_table_stats(Input,Output,Truth,epoch,MetricFun):
    # W&B: Create a Table to store predictions for each test step
    columns=["id", "image","Masks","Calculated Tissue score","Real Tissue score"]
    test_table = wandb.Table(columns=columns)
    # Y_pred_target=torch.argmax(Output,dim=1)
    Y_true_target=torch.argmax(Truth[0],dim=1).cpu().numpy()
    # metric_calc = MetricFun(Output.ravel(), Truth[1])
    # r2score = metric_calc["test_R2Score"].cpu().numpy()
    # mse = metric_calc["test_MeanSquaredError"].cpu().numpy()
    for i in range(16):
        idx = f"{epoch}_{i}"
        image = wandb.Image(Input[i].permute(1,2,0).cpu().numpy())

        mask = wandb.Image(image, masks={
            "Tissue mask": {"mask_data" : Y_true_target[i], "class_labels" : CLASSES},
        })
        # metric_calc = MetricFun(Output[i][0], Truth[1][i])
        test_table.add_data(idx, image,mask,Output[i][0],Truth[1][i])
    wandb.log({"table_key": test_table})

if __name__=="__main__":
    ###################################
    #          Model Setup            #
    ###################################
    NUM_CLASSES = 1
    DEVICE_LIST = data_hyp["DEVICE_LIST"]
    NUM_GPUS = len(DEVICE_LIST)
    CLASSES = {0:'invasive tumor',1:'tumor-associated stroma',2:'inflamed stroma',3:'rest'}
    device = torch.device(f"cuda:{DEVICE_LIST[0]}" if torch.cuda.is_available() else "cpu")

    if (transfer_load is not None) and (resume_location is None):
        print("Loading pretraining weights from previous experiments...")
        model = Resnet_bihead_v2(data_hyp["ENCODER_NAME"])
        model = load_model_weights(model, transfer_load,device)
    elif resume_location is not None:
        model = Resnet_bihead_v2(data_hyp["ENCODER_NAME"])
    else:
        model = Resnet_bihead_v2(data_hyp["ENCODER_NAME"],pretrained=True)

    #Freeze only cell layers
    for name, param in model.named_parameters():
        # if (("cell" in name) | ("model" in name)) and not ((("layer4") in name)|(("fc") in name)):
        # if ("cell" in name) | ("model" in name):
        if ("cell" in name):
            param.requires_grad = False

    #Multiple GPUs
    if torch.cuda.device_count() > 1  and multi_gpu:
        print("Using {} GPUs".format(NUM_GPUS))
        model = DataParallelModel(model, device_ids=DEVICE_LIST)
    
    model = model.to(device)

    ###################################
    #        Data Augmentations       #
    ###################################   
    
    # transform_train =  albumentations.Compose([    
    #     albumentations.Resize(256,256),
    #     albumentations.ToFloat(max_value=255),
    #     albumentations.OneOf([
    #         albumentations.VerticalFlip(p=0.5),
    #         albumentations.HorizontalFlip(p=0.5),              
    #         albumentations.Rotate(p=0.5)
    #         ],p=0.6),
    #     albumentations.OneOf([
    #         albumentations.GaussianBlur(p=0.4),
    #         # albumentations.RandomBrightnessContrast(p=0.4)
    #         # albumentations.Sharpen(p=0.4)
    #     ],p=1),
    #     # albumentations.HueSaturationValue(hue_shift_limit=0.6, sat_shift_limit=0.6, val_shift_limit=0.6,p=0.5),
    #     HSV(factor=0.25,p=0.5),
    #     # albumentations.Affine(scale=(0.8,1.2),translate_percent=(0,0.3),shear=(-7,7),p=0.4),
    #     ToTensorV2()])
    # transform_train =  albumentations.Compose([
    #     ToTensorV2()]) 
    transform_train =  albumentations.Compose([    
        albumentations.Resize(256,256),
        albumentations.ToFloat(max_value=255),
        albumentations.OneOf([
            albumentations.VerticalFlip(p=0.5),
            albumentations.HorizontalFlip(p=0.5),              
            albumentations.Rotate(p=0.5)
            ],p=0.4),
        albumentations.OneOf([
            albumentations.GaussianBlur(p=0.4),
            albumentations.RandomBrightnessContrast(p=0.4)
            # albumentations.Sharpen(p=0.4)
        ],p=0.5),
        # albumentations.HueSaturationValue(hue_shift_limit=0.6, sat_shift_limit=0.6, val_shift_limit=0.6,p=0.5),
        # HSV(factor=0.25,p=0.5),
        # albumentations.Affine(scale=(0.8,1.2),translate_percent=(0,0.3),shear=(-7,7),p=0.4),
        ToTensorV2()])
    
    transform_test =  albumentations.Compose([
        albumentations.Resize(256,256),
        albumentations.ToFloat(max_value=255),
        ToTensorV2()])
    ###################################
    #          Data Loaders           #
    ###################################  
    config_file = data_hyp['DATALOADER_CONFIG']

    ###########################################
    #  Loss function, optimizers and metrics  #
    ########################################### 
    
    #Loss function
    # LossFun = MultiFocalLoss(alpha=data_hyp['FOCAL_IMP'],gamma=data_hyp['FOCAL_GAMMA'])
    # LossFun = torch.nn.MSELoss()
    LossFun = torch.nn.L1Loss(reduction="sum")
    # LossFun = nn.CrossEntropyLoss().cuda(device)
    LossFun = LossFun.to(device)
    #optimizer: Only optimize tissue layer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=data_hyp['LEARNINGRATE'], weight_decay = data_hyp['LAMBDA'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=data_hyp["PATIENCE"])
    #Metrics
    MetricFun = torchmetrics.MetricCollection([torchmetrics.MeanAbsoluteError(dist_sync_on_step=multi_gpu),
                                            torchmetrics.R2Score(dist_sync_on_step=multi_gpu)])
    #                                         torchmetrics.F1Score(num_classes=NUM_CLASSES,average=None,dist_sync_on_step=multi_gpu)])
    # metrics = torchmetrics.MetricCollection([DiceScore(labels=[0,1,2],dist_sync_on_step=multi_gpu)])
    
    TrainMetricDict = MetricFun.clone(prefix='train_').to(device)
    TestMetricDict = MetricFun.clone(prefix='test_').to(device)
    # DiceScore = torchmetrics.functional.dice_score
    # dicescore = DiceScore(labels=[0,1,2])
    # dicescore = compute_meandice_multilabel
    best_metric = -10000

    if is_wandb:
        wandb.watch(model, log='all') 
    ###################################
    #           Training              #
    ###################################
    if resume_location is not None:
        model,optimizer,scheduler,epoch_start,best_metric = load_checkpoint(resume_location,model,optimizer,scheduler)
        #best_metric = best_metric.cpu().numpy()
        print("Resuming from saved checkpoint...")
    else:
        epoch_start = 0

    #Forcefully set LR
    optimizer.param_groups[0]['lr'] = data_hyp['LEARNINGRATE']   
    for epoch in range(epoch_start,data_hyp['EPOCHS']):
        print("EPOCH: {}".format(epoch))
        train(model, optimizer, LossFun,TrainMetricDict)
        metric,test_loss = test(model ,LossFun,epoch,TestMetricDict)
        ### Saving model checkpoints
        if is_save and (((epoch+1) % 3 == 0) or (metric>best_metric)):
            save_checkpoint(epoch,model,optimizer,scheduler,metric)
            best_metric = metric
        scheduler.step(test_loss)
        if is_wandb:
            wandb.log({"Learning rate":optimizer.param_groups[0]["lr"]})
    print("Training done...")
