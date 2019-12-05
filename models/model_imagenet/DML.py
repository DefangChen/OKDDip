import torch
import torch.nn as nn
from .resnet import * 
from .densenet import * 

__all__ = ['MutualNet']

class MutualNet(nn.Module):
    def __init__(self, model="resnet32", num_branches = 4, num_classes=10):
        super(MutualNet, self).__init__()
        self.num_branches = num_branches
        
        for i in range(num_branches):
            if model == "resnet34":
                setattr(self, 'stu'+str(i), resnet34(num_classes = num_classes))
            elif model == "densenetd40k12":
                setattr(self, 'stu'+str(i), densenetd40k12(num_classes = num_classes))
            
    def forward(self, x):
        out = self.stu0(x) 
        out = out.unsqueeze(-1)
        for i in range(1, self.num_branches):
            temp_out = getattr(self, 'stu'+str(i))(x)
            temp_out = temp_out.unsqueeze(-1)
            out = torch.cat([out, temp_out],-1)        
        return out