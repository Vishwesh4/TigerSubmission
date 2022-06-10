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
        self.model = torchvision.models.__dict__[model_name](pretrained=pretrained)
        
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

class Resnet_bihead(nn.Module):
    def __init__(self, model_name,pretrained=False):
        super(Resnet_bihead, self).__init__()
        self.model = torchvision.models.__dict__[model_name](pretrained=pretrained)
        
        #Cell Head
        self.cell = nn.Sequential(nn.Linear(self.model.fc.out_features,400),
                                  nn.Linear(400,200),
                                  nn.Linear(200,1),
                                  torch.nn.Sigmoid())

        #Tissue Head
        self.tissue = nn.Sequential(nn.Linear(self.model.fc.out_features,400),
                                    nn.Linear(400,200),
                                    nn.Linear(200,1),
                                    torch.nn.Sigmoid())

    def forward(self, image):
        feat = self.model(image)

        cell_score = self.cell(feat)
        tissue_score = self.tissue(feat)
        return cell_score,tissue_score

class NN_pooler(torch.nn.Module):
    def __init__(self,input_features):
        super(NN_pooler, self).__init__()
        self.fc1 = nn.Linear(input_features,5)
        # self.fc2 = nn.Linear(40,10)
        self.fc3 = nn.Linear(5,1)
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
    def forward(self, x):
        output = self.relu(self.fc1(x))
        # output = self.fc1(x)
        # output = self.relu(self.fc2(output))
        output = self.fc3(output)
        output = self.sigm(output)
        return output