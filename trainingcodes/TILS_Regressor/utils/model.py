import warnings
import torch
import torch._utils
import torch.nn.functional as F
import torch.nn as nn
import torchvision

class Resnet_bihead_v2(nn.Module):
    """
    Same as earlier model except it has dropout
    """
    def __init__(self, model_name,pretrained=False):
        super(Resnet_bihead_v2, self).__init__()
        self.model = torchvision.models.__dict__[model_name](pretrained=False)
        
        #Cell Head
        self.cell = nn.Sequential(nn.Linear(self.model.fc.out_features,400),
                                  nn.Dropout(p=0.15),
                                  nn.Linear(400,200),
                                  nn.Linear(200,1),
                                  torch.nn.Sigmoid())

        #Tissue Head
        self.tissue = nn.Sequential(nn.Linear(self.model.fc.out_features,400),
                                    # nn.Linear(400,400),
                                    nn.Dropout(p=0.15),
                                    nn.Linear(400,200),
                                    nn.Dropout(p=0.05),
                                    nn.Linear(200,1),
                                    nn.Dropout(p=0.05),
                                    torch.nn.Sigmoid())
    def forward(self, image,head="all"):
        feat = self.model(image)
        if head=="all":
            cell_score = self.cell(feat)
            tissue_score = self.tissue(feat)
            return cell_score,tissue_score
        elif head=="cell":
            return self.cell(feat)
        elif head=="tissue":
            return self.tissue(feat)
        else:
            raise ValueError("Incorrect head given, choose out of all/cell/tissue")

    def load_model_weights(self,path,device=torch.device("cpu")):
        """
        Loads model weight
        """
        state = torch.load(path, map_location=device)
        state_dict = state['model_state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('resnet.', '').replace('module.', '')] = state_dict.pop(key)
        model_dict = self.state_dict()   
        weights = {k: v for k, v in state_dict.items() if k in model_dict}
        if len(state_dict.keys())!=len(model_dict.keys()):
            warnings.warn("Warning... Some Weights could not be loaded")
        if weights == {}:
            warnings.warn("Warning... No weight could be loaded..")
        model_dict.update(weights)
        self.load_state_dict(model_dict)