import warnings
from tqdm import tqdm
from time import time

import numpy as np
import torch
import torchvision
import torch.nn as nn

from .utils import Resnet_bihead_v2, postprocess, TILData


class TILEstimator_areaindv:
    """
    Network 1, with til head (til density) and invasive tumor head
    """
    def __init__(self,Tumor_path:str,
                 TIL_path:str,
                 transform,
                 spacing:float,
                 biopsy:bool,
                 is_tta = False,
                 transform_tta = [],
                 threshold_tumor=0.5,
                 device=torch.device("cpu"),
                 pool_method="add") -> None:
        self.device = device
        self.transform = transform
        self.threshold_tumor = threshold_tumor
        self.pool_method = pool_method
        self.spacing = spacing
        self.tta = is_tta
        self.transform_tta = transform_tta
        self.biopsy = biopsy
        #TIL network
        TILNet = Resnet_bihead_v2("resnet34",pretrained=False)
        
        #Tumour model
        TumourNet = torchvision.models.__dict__["resnet18"](pretrained=False)
        TumourNet.fc = nn.Sequential(
                    torch.nn.Linear(TumourNet.fc.in_features, 300),
                    torch.nn.Linear(300, 2),
                    torch.nn.Softmax(-1)
                    )
        
        #Load model weights
        self.TumourNet = self.load_model_weights(TumourNet,Tumor_path).to(device)
        self.TILNet = self.load_model_weights(TILNet,TIL_path).to(device)
        self.TumourNet.eval()
        self.TILNet.eval()

        #For storing patch results for pooling
        self.reset()
    
    def load_model_weights(self,model,model_path):
        """
        Loads model weight
        """
        state = torch.load(model_path, map_location=self.device)
        state_dict = state['model_state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('resnet.', '').replace('module.', '')] = state_dict.pop(key)
        model_dict = model.state_dict()   
        weights = {k: v for k, v in state_dict.items() if k in model_dict}
        if len(state_dict.keys())!=len(model_dict.keys()):
            warnings.warn("Warning... Some Weights could not be loaded")
        if weights == {}:
            warnings.warn("Warning... No weight could be loaded..")
        model_dict.update(weights)
        model.load_state_dict(model_dict)
        return model

    def collect(self,img):
        if not self.tta:
            img = self.transform(img)
        self.allpatch.append(img)

    def get_template(self,template):
        self.template = template

    def _form_dataset(self):
        if self.tta:
            return TILData(self.allpatch,transform_tta=self.transform_tta)
        else:
            return self.allpatch
    
    def timeit(self):
        self.time.append(time())

    def calculate_chunk(self):
        self.timeit()
        dataset = self._form_dataset()
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=64)
        chunk_results = np.empty((3,0),np.float64)
        with torch.no_grad():
            for data in tqdm(dataloader, desc=f"Processing chunk {len(self.time)}"):
                images = data.to(self.device)
                
                if self.tta:
                    n,a,c,h,w = images.shape
                    images = torch.reshape(images,(n*a,c,h,w))
                is_relevant = self.TumourNet(images)[:,-1]
                tilscore,tissuescore = self.TILNet(images)
                if self.tta:
                    # Test time augmentation mean of outputs of augmentation
                    is_relevant = torch.mean(is_relevant.reshape(-1,a),1)
                    tilscore = torch.mean(tilscore.reshape(-1,a),1)
                    tissuescore = torch.mean(tissuescore.reshape(-1,a),1)
                # raise ValueError

                results = np.stack((is_relevant.cpu().numpy(),tilscore.cpu().ravel().numpy(),tissuescore.cpu().ravel().numpy()))
                chunk_results = np.hstack((chunk_results,results))
        self.allpatch_results.append(chunk_results)
        #Reset to save memory
        self.allpatch = []
        del dataset, chunk_results

    def compute_til(self):
        #Collect all results
        if len(self.allpatch_results)==1:
            self.allpatch_results = self.allpatch_results[0]
        else:
            self.allpatch_results = np.hstack(self.allpatch_results)
        #Post processing of results
        processed_tumorbed,processed_tissue = postprocess(self.template,self.spacing,self.allpatch_results[0,:],self.allpatch_results[-1,:],self.biopsy)
        self.allpatch_results[0,:] = processed_tumorbed
        self.allpatch_results[-1,:] = processed_tissue

        if self.pool_method == "add":
            return self._pool_results()
        else:
            raise NotImplementedError("Pooling method not implemented, choose out of add or nn")

    def _pool_results(self):
        idx = np.where(self.allpatch_results[0,:]>=self.threshold_tumor)[0]
        X_til = self.allpatch_results[:,idx]
        total_til = np.sum(X_til[1,:])
        total_tissues = np.sum(X_til[2,:])
        calc = 100*total_til/(total_tissues+0.00001)
        calc = np.clip(calc,a_min=0,a_max=100)
        self.timeit()
        print("func: TIL score took %2.4f sec" % (self.time[-1]-self.time[0]))
        return calc
    
    @staticmethod
    def isforeground(patch,threshold=0.95):
        assert len(patch.shape)==2
        h,w = patch.shape
        return np.sum(patch)/float(h*w)>threshold
    
    def reset(self):
        self.allpatch = []
        self.template_list = []
        self.row_list = []
        self.count = 0
        self.time = []
        self.allpatch_results = []
        # self.chunk_results = np.empty((3,0),np.float64)
