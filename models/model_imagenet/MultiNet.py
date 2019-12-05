import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import *
from .densenet import *

__all__ = ['StuNet']

class StuNet(nn.Module):

    def __init__(self, model="resnet32", num_branches = 4, num_classes=10, input_channel=64, en = False, factor=8):
        super(StuNet, self).__init__()
        self.num_branches = num_branches
        self.en = en
        for i in range(num_branches):
            if model == "resnet34":
                setattr(self, 'stu'+str(i), resnet34(num_classes = num_classes, KD = True))
            elif model == "densenetd40k12":
                setattr(self, 'stu'+str(i), densenetd40k12(num_classes = num_classes, KD = True))
            
        self.query_weight = nn.Linear(input_channel, input_channel//factor, bias = False)
        self.key_weight = nn.Linear(input_channel, input_channel//factor, bias = False)
            
    def forward(self, x):
        x_f, pro = self.stu0(x)                         # (B X 64), (B X num_classes)
        proj_query = self.query_weight(x_f)             # B x input_channel//factor
        proj_key = self.key_weight(x_f)                 # B x input_channel//factor
         
        proj_query = proj_query[:, None, :]
        proj_key = proj_key[:, None, :]
        pro = pro.unsqueeze(-1)
        if self.en:
            for i in range(1, self.num_branches):
                temp_x_f, temp_pro = getattr(self, 'stu'+str(i))(x)
                x_f_q = self.query_weight(temp_x_f)                # B x input_channel//factor
                x_f_k = self.key_weight(temp_x_f)                  # B x input_channel//factor
                temp_pro = temp_pro.unsqueeze(-1)
                # B X num_students X input_channel//factor
                proj_query = torch.cat([proj_query, x_f_q[:,None,:]], 1) 
                proj_key = torch.cat([proj_key, x_f_k[:,None,:]], 1) 
                # B X num_classes X num_students
                pro = torch.cat([pro, temp_pro],-1)          
                # B X num_students X num_features
            energy = torch.bmm(proj_query, proj_key.permute(0,2,1)) 
            attention = F.softmax(energy, dim = -1) 
            x_m = torch.bmm(pro, attention.permute(0,2,1))
            return pro, x_m
        else:
            for i in range(1, self.num_branches-1):
                temp_x_f, temp_pro = getattr(self, 'stu'+str(i))(x)
                x_f_q = self.query_weight(temp_x_f)                # B x input_channel//factor
                x_f_k = self.key_weight(temp_x_f)                  # B x input_channel//factor
                temp_pro = temp_pro.unsqueeze(-1)
                # B X num_students X input_channel//factor
                proj_query = torch.cat([proj_query, x_f_q[:,None,:]], 1) 
                proj_key = torch.cat([proj_key, x_f_k[:,None,:]], 1) 
                # B X num_classes X num_students
                pro = torch.cat([pro, temp_pro],-1)          
            energy = torch.bmm(proj_query, proj_key.permute(0,2,1)) 
            attention = F.softmax(energy, dim = -1) 
            x_m = torch.bmm(pro, attention.permute(0,2,1))
            
            _, temp_pro = getattr(self, 'stu'+str(self.num_branches - 1))(x)
            
            return pro, x_m, temp_pro
            