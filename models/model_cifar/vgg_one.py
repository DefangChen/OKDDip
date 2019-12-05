'''
VGG16 for CIFAR-10/100 Dataset.

Reference:
1. https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
2. NIPS18-Knowledge Distillation by On-the-Fly Native Ensemble
3. NIPS18-Collaborative Learning for Deep Neural Networks

'''

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['vgg16', 'vgg19']

#cfg = {
#    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
#}

class ILR(torch.autograd.Function):
   
    @staticmethod
    def forward(ctx, input, num_branches):
        ctx.num_branches = num_branches
        return input

    @staticmethod
    def backward(ctx, grad_output):
        num_branches = ctx.num_branches
        return grad_output/num_branches, None


class VGG(nn.Module):
    def __init__(self, num_classes=10, num_branches=3, bpscale = False, avg = False, ind = False, depth=16):
        super(VGG, self).__init__()
        self.inplances = 64
        self.avg = avg
        self.bpscale = bpscale
        self.num_branches = num_branches
        self.conv1 = nn.Conv2d(3, self.inplances, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.inplances)
        self.conv2 = nn.Conv2d(self.inplances, self.inplances, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.inplances)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ind = ind
        
        self.layer1 = self._make_layers(128, 2)
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
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),))
            
        if self.avg == False:
            self.avgpool_c = nn.AdaptiveAvgPool2d((1,1))
            self.control_v1 = nn.Linear(self.inplances, self.num_branches)
            self.bn_v1 = nn.BatchNorm1d(self.num_branches)
        if self.bpscale:
            self.layer_ILR = ILR.apply
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
                nn.init.constant_(m.bias, 0)
    
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
        if self.bpscale:
            x = self.layer_ILR(x, self.num_branches) # Backprop rescaling
            
        x_3 = getattr(self,'layer3_0')(x)   # B x 64 x 8 x 8
        x_3 = x_3.view(x_3.size(0), -1)     # B x 64
        x_3_1 = getattr(self, 'classifier3_0')(x_3)     # B x num_classes
        pro = x_3_1.unsqueeze(-1)        
        for i in range(1, self.num_branches):
            temp = getattr(self, 'layer3_'+str(i))(x)
            temp = temp.view(temp.size(0), -1)   
            temp_1 = getattr(self, 'classifier3_' + str(i))(temp)
            temp_1 = temp_1.unsqueeze(-1)
            pro = torch.cat([pro,temp_1],-1)        # B x num_classes x num_branches
        
        if self.ind:
            return pro, None
        # CL
        else:
            if self.avg:
                x_m = 0
                for i in range(1, self.num_branches):
                    x_m += 1/(self.num_branches-1) * pro[:,:,i]
                x_m = x_m.unsqueeze(-1)
                for i in range(1, self.num_branches):
                    temp = 0
                    for j in range(0, self.num_branches):
                        if j != i:
                            temp += 1/(self.num_branches-1) * pro[:,:,j]       # B x num_classes
                    temp = temp.unsqueeze(-1)
                    x_m = torch.cat([x_m, temp],-1)                            # B x num_classes x num_branches
            # ONE
            else:
                x_c=self.avgpool_c(x)
                x_c = x_c.view(x_c.size(0), -1) # B x 32 
                x_c=self.control_v1(x_c)    # B x 3
                x_c=self.bn_v1(x_c)  
                x_c=F.relu(x_c)      
                x_c = F.softmax(x_c, dim=1) # B x 3  
            
                x_3 = getattr(self,'layer3_0')(x)   # B x 64 x 8 x 8
                x_3 = x_3.view(x_3.size(0), -1)     # B x 64
                x_3_1 = getattr(self, 'classifier3_0')(x_3)     # B x num_classes
                x_m = x_c[:,0].view(-1, 1).repeat(1, x_3_1.size(1)) * x_3_1
                pro = x_3_1.unsqueeze(-1) 
                for i in range(1, self.num_branches):
                    temp = getattr(self, 'layer3_'+str(i))(x)
                    temp = temp.view(temp.size(0), -1)   
                    temp_1 = getattr(self, 'classifier3_' + str(i))(temp)
                    x_m += x_c[:,i].view(-1, 1).repeat(1, temp_1.size(1)) * temp_1       # B x num_classes
                    temp_1 = temp_1.unsqueeze(-1)
                    pro = torch.cat([pro,temp_1],-1)        # B x num_classes x num_branches
              
            return pro, x_m
    
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
