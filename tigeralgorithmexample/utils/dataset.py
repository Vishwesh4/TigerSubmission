"""
This script contains loading of list into dataset for inference. Mainly made for test time augmentations
Created 24th May
"""
from typing import List
import numpy as np
import torch
import albumentations
import torch.utils.data as data

class TILData(data.Dataset):
    ''' Implemented for getting the dataset and performing Test time augmentation during inference'''
    def __init__(self,
                data_list:List[np.array],
                transform_tta:List[albumentations.ImageOnlyTransform] = [],
                transform_default =  albumentations.ToFloat(max_value=255)):
        '''
        data_file: (pd.DataFrame) Contains all the file names and TIL density scoring
        path: (str) Path to TIL density dataset
        transform: (albumentations.core.transforms) Transform object
        labels: What all labels to consider for tissue segmentation
        labels_tils: 
        '''
        super(TILData,self).__init__()
        self.data_list = data_list
        self.transform_tta = transform_tta
        self.transform = transform_default
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self,index):
        img = self.data_list[index]
        img_proc = self.transform(image=img)["image"]
        return self._get_tta(img_proc)

    def _get_tta(self,img):
        """
        Given an image, performs test time augmentation
        """
        img_trans_list = [img]
        for transforms in self.transform_tta:
            img_trans = transforms(image=img)["image"]
            img_trans_list.append(img_trans)
        return torch.Tensor(np.array(img_trans_list)).permute(0,3,1,2)
