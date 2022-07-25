import torch.utils.data as data
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import random

class Tumourbed_dataset(data.Dataset):
    ''' Gets the tissue, cell and til density score in a patch'''
    def __init__(self,
                data_path:str="/localdisk3/ramanav/Tumourbed_patches",
                transform=None):
        '''
        data_file: (pd.DataFrame) Contains all the file names and TIL density scoring
        path: (str) Path to TIL density dataset
        transform: (albumentations.core.transforms) Transform object
        labels: What all labels to consider for tissue segmentation
        labels_tils: 
        '''
        super(Tumourbed_dataset,self).__init__()
        self.data_path = Path(data_path)
        self.transform = transform
        if not (self.data_path/"all_paths.csv").is_file():
            self.df = self._get_all_paths()
        else:
            self.df = pd.read_csv(self.data_path/"all_paths.csv")
    
    def __len__(self):
        return len(self.df)
    
    def _get_all_paths(self):
        '''
        Gets all the paths and stores it into all_paths.csv
        '''
        folders = [Path("negetive_images"),
                   Path("negetive_images_1"),
                   Path("negetive_images_2"),
                   Path("negetive_images_3"),
                   Path("negetive_images_4"),
                   Path("negetive_images_5")]
        data = []
        print("Getting all the paths of samples in the folder...")
        for folder in folders:
            folder_files =  (self.data_path/folder).glob("*.npy")
            data.extend(folder_files)
        df = pd.DataFrame(data={"Name":data})
        df.to_csv(self.data_path/"all_paths.csv",index=False)
        return df
    
    def __getitem__(self,index):
        metafile = self.df.iloc[index]

        #Loading images
        img = np.load(metafile.Name)
        img = img.astype(np.float32)/255

        if self.transform!=None:
            img_processed = self.transform(img)
            return img_processed,index
        else:
            return img
    
    
class Tumourbed_dataset_train(data.Dataset):
    ''' Gets the tissue, cell and til density score in a patch'''
    def __init__(self,
                train_split:float,
                mode:str,
                data_path:str="/localdisk3/ramanav/Tumourbed_training",
                transform=None,
                seed=2022):
        '''
        data_path: (str) Path to TIL density dataset
        transform: (torchvision.transforms) Transform object
        mode: (str) training/validation
        train_split: (float) between 0-1
        '''
        super(Tumourbed_dataset_train,self).__init__()
        self.mode = mode
        self.train_split = train_split
        self.data_path = Path(data_path)
        self.transform = transform
        if not (self.data_path/"Detection_dataset_v3.csv").is_file():
            self.df = self._form_dataset()
        else:
            # self.df = pd.read_csv(self.data_path/"Tumourbed_patches/Detection_dataset.csv").sample(frac=1,random_state=seed)
            self.df = pd.read_csv(self.data_path/"Detection_dataset_v3.csv").sample(frac=1,random_state=seed)
            # self.df["class"] = self.df["class"].astype(int)
        if mode=="training":
            self.df = self.df.iloc[:int(self.train_split*len(self.df)),:].reset_index(drop=True)
        else:
            self.df = self.df.iloc[int(self.train_split*len(self.df)):,:].reset_index(drop=True)

    def __len__(self):
        return len(self.df)
    
    def _form_dataset(self):
        #Mix both the neg. samples and tissue segmentation dataset
        #Positive samples
        self.data_path = self.data_path.parent
        tissue_seg = pd.read_csv(self.data_path/Path("Seg_Patches/img_locations_training.csv"))
        tissue_seg["area_prop"] = (tissue_seg["class_1"]+tissue_seg["class_2"]+tissue_seg["class_6"])/(256.0*256.0)
        #Get 100k examples of positive samples with area of important regions >0.25
        tissue_seg = tissue_seg.loc[tissue_seg["area_prop"]>=0.25].sample(frac=0.5).reset_index(drop=True)
        tissue_seg = tissue_seg.drop(columns=["class_1","class_2","class_3","class_4","class_5","class_6","class_7","area_prop"])
        tissue_seg["class"] = 1
        tissue_seg["cluster_id"] = np.nan
        #Negative samples
        with open(str(self.data_path/Path('Tumourbed_patches/cluster_sample.pickle')), 'rb') as handle:
            data = pickle.load(handle)      
        negfil_loc = pd.read_csv(self.data_path/"Tumourbed_patches/all_paths.csv")
        for i in range(len(data)):
            file_locs = list(negfil_loc.loc[data[i]["indices"],"Name"])
            clusterids = list(i*np.ones(len(file_locs)))
            classes = list(np.zeros(len(file_locs)))
            tissue_seg = tissue_seg.append(pd.DataFrame(data={"Name":file_locs,"class":classes,"cluster_id":clusterids}),ignore_index=True)
        tissue_seg.to_csv(self.data_path/"Tumourbed_patches/Detection_dataset.csv",index=False)
        return tissue_seg

    def __getitem__(self,index):
        metafile = self.df.iloc[index]
        label =  metafile["class"]
        path = metafile["Name"]
        #Loading images
        # if label==0:
        #     img = np.load(path).astype(np.uint8)
        # else:
        #     img = np.load(str(self.data_path/Path("Seg_Patches")/path)).astype(np.uint8)
        img = np.load(str(self.data_path/path)).astype(np.uint8)
        
        # img = img.astype(np.float32)/255

        if self.transform is not None:
            img_processed = self.transform(img)
            return img_processed,label
        else:
            return img,label