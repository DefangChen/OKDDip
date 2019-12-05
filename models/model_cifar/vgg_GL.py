'''
VGG16 for CIFAR-10/100 Dataset.

Reference:
1. https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
__all__ = ['vgg16', 'vgg19']

#cfg = {
#    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
#}

class VGG(nn.Module):
    def __init__(self, num_classes=10, num_branches=3, factor = 8, en= False, depth=16, dropout = 0.5):
        super(VGG, self).__init__()
        self.inplances = 64
        self.en = en
        self.num_branches = num_branches
        self.conv1 = nn.Conv2d(3, self.inplances, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.inplances)
        self.conv2 = nn.Conv2d(self.inplances, self.inplances, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.inplances)
        self.relu = nn.ReLU(inplace=True)            
        self.layer1 = self._make_layers(128, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        if depth == 16:
            num_layer = 3
        elif depth == 19:
            num_layer = 4
        
        self.layer2 = self._make_layers(256, num_layer)
        self.layer3 = self._make_layers(512, num_layer)
        for i in range(num_branches):
            setattr(self, 'layer3_'+str(i), self._make_layers(512, num_layer))
            
            setattr(self, 'classifier3_'+str(i), nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p = dropout),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p = dropout),
            nn.Linear(512, num_classes),
            ))
    
        input_channel = 512
        self.query_weight = nn.Linear(input_channel, input_channel//factor, bias = False)
        self.key_weight = nn.Linear(input_channel, input_channel//factor, bias = False)        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
    
    def _make_layers(self, input, num_layer):    
        layers=[]
        for i in range(num_layer):
            conv2d = nn.Conv2d(self.inplances, input, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(input), nn.ReLU(inplace=True)]
            self.inplances = input
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
    
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_3 = getattr(self,'layer3_0')(x)   # B x 512 x 1 x 1
        x_3 = x_3.view(x_3.size(0), -1)     # B x 512
        proj_q = self.query_weight(x_3)     # B x 64
        proj_q = proj_q[:, None, :]
        proj_k = self.key_weight(x_3)       # B x 64 
        proj_k = proj_k[:, None, :]
        x_3_1 = getattr(self, 'classifier3_0')(x_3)     # B x num_classes
        pro = x_3_1.unsqueeze(-1)        
        if self.en:
            for i in range(1, self.num_branches):
                temp = getattr(self, 'layer3_'+str(i))(x)
                temp = temp.view(temp.size(0), -1)  
                temp_q = self.query_weight(temp)
                temp_k = self.key_weight(temp)
                temp_q = temp_q[:, None, :]
                temp_k = temp_k[:, None, :]
                temp_1 = getattr(self, 'classifier3_' + str(i))(temp)
                temp_1 = temp_1.unsqueeze(-1)
                pro = torch.cat([pro,temp_1],-1)        # B x num_classes x num_branches
                proj_q = torch.cat([proj_q, temp_q], 1) # B x num_branches x 8
                proj_k = torch.cat([proj_k, temp_k], 1) 
            
            energy = torch.bmm(proj_q, proj_k.permute(0,2,1)) 
            attention = F.softmax(energy, dim = -1) 
            x_m = torch.bmm(pro, attention.permute(0,2,1))
            return pro, x_m
        else:
            for i in range(1, self.num_branches - 1):
                temp = getattr(self, 'layer3_'+str(i))(x)
                temp = temp.view(temp.size(0), -1)   
                temp_q = self.query_weight(temp)
                temp_k = self.key_weight(temp)
                temp_q = temp_q[:, None, :]
                temp_k = temp_k[:, None, :]
                temp_1 = getattr(self, 'classifier3_' + str(i))(temp)
                temp_1 = temp_1.unsqueeze(-1)
                pro = torch.cat([pro,temp_1],-1)        # B x num_classes x num_branches
                proj_q = torch.cat([proj_q, temp_q], 1) # B x num_branches x 8
                proj_k = torch.cat([proj_k, temp_k], 1) 
            
            energy =  torch.bmm(proj_q, proj_k.permute(0,2,1)) 
            attention = F.softmax(energy, dim = -1) 
            x_m = torch.bmm(pro, attention.permute(0,2,1))
            
            temp = getattr(self, 'layer3_'+str(self.num_branches - 1))(x)
            temp = temp.view(temp.size(0), -1)   
            temp_out = getattr(self, 'classifier3_' + str(self.num_branches - 1))(temp)
            return pro, x_m, temp_out

def vgg16(pretrained=False, path=None, **kwargs):
    """
    Constructs a VGG16 model.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    model = VGG(depth=16, **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model
    
def vgg19(pretrained=False, path=None, **kwargs):
    """
    Constructs a VGG19 model.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    model = VGG(depth=19, **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model
