# import torch
# import utils.dataset_cls as ds
# from torch.utils import data
# import os
# from utils import preprocessing
# import models.models as models
# import torchvision.models.resnet as resnet
# import numpy as np
# from tqdm import tqdm
# import random
# from sklearn.cluster import MiniBatchKMeans as KMeans
# from sklearn.cluster import *
# import torchvision
# from PIL import Image
# from sklearn.neighbors import NearestNeighbors
# from scipy.stats import mode
# from sklearn import svm
# from sklearn import linear_model
# from sklearn.svm import LinearSVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn import tree
# from efficientnet_pytorch import EfficientNet

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import torchvision
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import random
import torch
import os
from utils.dataloader import Tumourbed_dataset
from pathlib import Path
import argparse
import pandas as pd
from utils.parallel import DataParallelModel, DataParallelCriterion, gather
import torch.nn as nn
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score
import pickle

# parser = argparse.ArgumentParser()
# parser.add_argument("-n", help="project name",required=True)
#For compute canada
# parser.add_argument("-l",help="Data location modfier",default=None)
# parser.add_argument("-t",help="Path for loading weights for transfer learning",default=None)

# args = parser.parse_args()
# loc_mod = args.l
# name = args.n
name = "neg_sample_exp1"
transfer_load = "/home/ramanav/tiger_project/tigeralgorithmexample/modules/TILdetection/Results/SSL/best_model_271.pt"

dataset_path = Path("/localdisk3/ramanav/Tumourbed_patches")

PARENT_NAME = Path("/home/ramanav/tiger_project/tigeralgorithmexample/modules/Tumourbed")
FOLDER_NAME = PARENT_NAME / Path("Results")
MODEL_SAVE = FOLDER_NAME / Path(name)


if not MODEL_SAVE.is_dir():
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

def calc_variance(kmeans_distances,labels,SST):
    tot_withinss = sum([d[labels[i]]**2 for i,d in enumerate(kmeans_distances)])
    SS_between = SST-tot_withinss
    return SS_between/float(SST)

def get_clusters(num_clusters,model,dataloader):
    '''
    Get clusters for the negative sampling
    '''
    with torch.no_grad():
        all_features = []
        indices = []
        if (dataset_path/"feat_vectors.npy").is_file():
            print("Loading feature vectors...")
            X = np.load(str(dataset_path/"feat_vectors.npy"))
            indices = np.load(str(dataset_path/"indices.npy"))
            print("Done")
            # X = np.random.rand(10000,15)
        else:
            print("Storing feature vectors...")
            for data in tqdm(dataloader):
                inputs,index = data
                #Convert to GPU
                inputs = inputs.to(device)
                outputs = model(inputs)
                if torch.cuda.device_count() > 1:
                    outputs = gather(outputs,device)
                all_features.extend(outputs.cpu())
                indices.extend(index.numpy())
            X = torch.stack(all_features).numpy()
            #Saving the feature vectors
            with open(str(dataset_path/"feat_vectors.npy"), 'wb') as f:
                np.save(f,X)
            with open(str(dataset_path/"indices.npy"), 'wb') as f:
                np.save(f,indices)
        # SST = sum(pdist(X)**2)/X.shape[0] 
        exp_var = []
        LABELS = []
        KMEANS = []
        for n_cls in num_clusters:
            print(f"cluster number : {n_cls}")
            clustering_obj = MiniBatchKMeans(n_clusters=n_cls, random_state=0).fit(X)
            kmeans_distances = clustering_obj.fit_transform(X)
            labels = clustering_obj.labels_
            # exp_var.append(calc_variance(kmeans_distances,labels,SST))
            exp_var.append(silhouette_score(X,labels,sample_size=50000,random_state=2022))
            LABELS.append(labels)
            KMEANS.append(kmeans_distances)
        print(f"Silhouette score for selected clusters({num_clusters}): {exp_var}")
        max_exp = np.argmax(exp_var)
        print(f"Selecting and saving results of cluster: {num_clusters[max_exp]}")
        return indices,LABELS[max_exp],KMEANS[max_exp],num_clusters[max_exp]

def sample_from_cluster(indices,labels,num_sample_per_cluster,kmeans_distance):
    data = []
    for cluster in np.unique(labels):
        meta = {}
        meta["cluster_id"] = cluster
        index = list(np.where(labels == cluster)[0])
        if len(index) >= 1:
            'randomly select from each cluster'
            # index = random.sample(index, num_sample_per_cluster)
            'take the most representative samples'
            index = np.argsort(kmeans_distance[:, cluster])[::5][:num_sample_per_cluster]
            meta["indices"] = indices[index]
        else:
            meta["indices"] = np.nan
        data.append(meta)   
    print("Saving the indices for each cluster...")
    # Store data (serialize)
    with open(str(dataset_path/'cluster_sample.pickle'), 'wb') as handle:
        pickle.dump(data, handle)

if __name__ == "__main__":
    #Model generate
    DEVICE_LIST = [3,4,5]
    NUM_GPUS = len(DEVICE_LIST)
    device = torch.device(f"cuda:{DEVICE_LIST[0]}" if torch.cuda.is_available() else "cpu")
    print("Starting neg sampling")
    if not (dataset_path/"feat_vectors.npy").is_file():
        if transfer_load is not None:
            print("Loading pretraining weights from previous experiments...")
            model = torchvision.models.__dict__["resnet18"](pretrained=False)
            model = load_model_weights(model, transfer_load)
        else:
            model = torchvision.models.__dict__["resnet18"](pretrained=True)

        #Multiple GPUs
        if torch.cuda.device_count() > 1:
            print("Using {} GPUs".format(NUM_GPUS))
            model = DataParallelModel(model, device_ids=DEVICE_LIST)
        
        model = model.to(device)

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        #Dataloader
        trainset = Tumourbed_dataset(data_path=dataset_path,transform=transform)
        dataloader = torch.utils.data.DataLoader(trainset, batch_size=512,shuffle=True,num_workers=4,pin_memory=True)
    else:
        model = None
        dataloader = None
    indices,labels,kmeans_distance,cluster_id = get_clusters([2500],model,dataloader)
    sample_from_cluster(indices,labels,int(100000/cluster_id),kmeans_distance)
    # indices,labels,kmeans_distance,cluster_id = get_clusters([100],model,dataloader)
