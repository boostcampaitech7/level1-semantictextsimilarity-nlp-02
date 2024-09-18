import torch.nn as nn 

def get_criterion(criterion_name) :
    if criterion_name == "MSE" :
        return nn.MSELoss()