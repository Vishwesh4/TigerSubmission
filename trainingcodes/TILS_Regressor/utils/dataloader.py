import torch.utils.data as data
import torch
from tqdm import tqdm
import cv2
from pathlib import Path
import pandas as pd
import numpy as np
import albumentations
from joblib import Parallel, delayed
from multiprocessing import Manager
from typing import Tuple

def process_input_bihead(x_batch:np.ndarray,y_batch:np.ndarray,transform:albumentations.core.composition.Compose,labels:list=[],labels_dens:list=[]):
    """
    Function for processing and performing data augmentation on the input data stream from wholeslidedata loader
    Parameters:
        labels(list): Only select certain labels, to form new one hot encoding
    Returns tuple of input(target,context) and output(target,context) data
    """
    x = []
    y = []

    x_batch1 = x_batch.astype(np.float32)
    y_batch1 = y_batch.astype(np.int32)
    if len(labels)!=0:
        y_batch1= y_batch1[:,:,:,labels]
        #If the first three classes are not present i.e 0,1,5 then put it into rest class which is at the end
        y_batch1[:,:,:,-1] = y_batch1[:,:,:,-1] + np.abs(np.sum(y_batch1,axis=-1)-1)
    for i in range(x_batch.shape[0]):
        y_batch1_target = [y_batch1[i,:,:,j] for j in range(y_batch1.shape[-1])]
        out = transform(image= x_batch1[i],masks=y_batch1_target) 
        x.append(out["image"])
        y.append(torch.Tensor(np.stack(out["masks"]))) 

    img = torch.stack(x)
    masks = torch.stack(y)

    #Calculate tissue density of given tissues
    h,w =  np.shape(masks[0,0,:,:])
    tissue_area = torch.sum(masks[:,labels_dens,:,:],dim=(1,2,3))/float(h*w)

    
    return img,[masks,tissue_area]

class TILS_dataset_Bihead_Area_Old(data.Dataset):
    ''' Gets the relevant tissue area and cell area in relevant tissue in a patch'''
    def __init__(self,
                data_file:pd.DataFrame,
                labels:list=[],
                labels_tils:list=[],
                path:str="/localdisk3/ramanav/TIL_Patches_v2",
                transform=None,
                transform_tta:list = []):
        '''
        data_file: (pd.DataFrame) Contains all the file names and TIL density scoring
        path: (str) Path to TIL density dataset
        transform: (albumentations.core.transforms) Transform object
        labels: What all labels to consider for tissue segmentation
        labels_tils: 
        '''
        super(TILS_dataset_Bihead_Area_Old,self).__init__()
        self.df = data_file.copy()
        self.path = Path(path)
        self.transform = transform
        self.labels = labels
        self.labels_tils = labels_tils
        self.transform_tta = transform_tta
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,index):
        metafile = self.df.iloc[index]
        IMGDIR = ["images","labels"]
        if (Path(metafile.Name).name[0]=="3") or (Path(metafile.Name).name[0]=="4"):
            IMGDIR = ["images_weight","labels_weight"]
        img_path = Path(self.path/Path(IMGDIR[0])/Path(metafile.Name))
        #Of dimension 2xHxW where dim=0 is the tissue segmentation mask and dim=1 is cell segmentation
        mask_path = Path(self.path/Path(IMGDIR[1])/Path(metafile.Name))

        #Loading images and labels
        img = np.load(img_path)
        labels = np.load(mask_path)
        tissue_mask = self._apply_onehot(labels[0])
        cell_mask = labels[1]

        if self.transform!=None:
            img_processed,all_mask = self.process_input(img,tissue_mask,cell_mask)
            if len(self.labels_tils)==0:
                return img_processed,all_mask[:-1]
            til_density,tissue_density = self._calc_area(all_mask)
            if len(self.transform_tta)>1:
                return self._get_tta(img_processed),(all_mask,til_density,tissue_density)
            else:
                return img_processed,(all_mask,til_density,tissue_density)
        else:
            #Will change that later
            return img,(np.stack((tissue_mask,cell_mask)),metafile.TILdensity)
    
    @staticmethod
    def _apply_onehot(mask):
        """
        Converts tissue mask into one hot encoding
        """
        y = mask
        #Replace roi class with rest
        y = np.where(y==0,7,y)
        y = y-1
        y_onehot = np.zeros((y.size,7))
        y_onehot[np.arange(y.size), y.ravel().astype(np.int32)] = 1
        y_onehot.shape = y.shape + (7,)
        return y_onehot

    def process_input(self,img:np.ndarray,tissue_mask:np.ndarray,cell_mask:np.ndarray):
        """
        Function for processing and performing data augmentation on the input data. Expects one hot form of tissue mask
        Parameters:
            labels(list): Only select certain labels, to form new one hot encoding
        Returns img and tissue+cell masks with the end dimension signifying cell mask
        """
        img = img.astype(np.float32)
        tissue_mask = tissue_mask.astype(np.int32)
        cell_mask = cell_mask.astype(np.int32)

        if len(self.labels)>1:
            tissue_mask = tissue_mask[:,:,self.labels]
            #If the first three classes are not present i.e 0,1,5 then put it into rest class which is at the end
            tissue_mask[:,:,-1] = tissue_mask[:,:,-1] + np.abs(np.sum(tissue_mask,axis=-1)-1)
        
        tissue_target = [tissue_mask[:,:,j] for j in range(tissue_mask.shape[-1])]
        tissue_target.append(cell_mask)
        out = self.transform(image= img,masks=tissue_target) 

        return out["image"],torch.Tensor(np.stack(out["masks"]))

    # def _calc_tils(self,masks):
    #     """
    #     Calculates tils density for the given transformed mask
    #     """
    #     TILS_area = 0
    #     tissue_area = 0
    #     # for k in range(len(self.labels_tils)):
    #     for k in self.labels_tils:
    #         tissue_area += torch.sum(masks[k,:,:])
    #         TILS_area += torch.sum(torch.logical_and(masks[-1,:,:],masks[k,:,:]))
    #     return TILS_area/(tissue_area+0.00001)
    def _get_tta(self,img):
        """
        Given an image, performs test time augmentation
        """
        img_trans_list = [img]
        for transforms in self.transform_tta:
            img_trans = transforms(image=img)["image"]
            img_trans_list.append(img_trans)
        return torch.Tensor(np.array(img_trans_list)).permute(0,3,1,2)

    def _calc_area(self,masks):
        """
        Calculates tils area in relevant tissue regions with tissue area as well for the given transformed mask
        """
        # cell_area = torch.sum(mask)
        # h,w =  np.shape(mask)
        # return torch.clamp(cell_area/(self.normalization_factor*float(h*w)),max=1.0)
        h,w = np.shape(masks[-1,:,:])
        TILS_area = 0
        tissue_area = 0
        # for k in range(len(self.labels_tils)):
        # TILS_area = torch.sum(masks[-1,:,:])/float(h*w)
        # tissue_area = torch.sum(masks[self.labels_tils,:,:])/float(h*w)
        # return TILS_area,tissue_area
        for k in self.labels_tils:
            tissue_area += torch.sum(masks[k,:,:])
            TILS_area += torch.sum(torch.logical_and(masks[-1,:,:],masks[k,:,:]))
        return TILS_area/float(h*w), tissue_area/float(h*w)