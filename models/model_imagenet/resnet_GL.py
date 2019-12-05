'''
ResNet for CIFAR-10/100 Dataset.

Reference:
1. https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
2. https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua
3. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
Deep Residual Learning for Image Recognition. https://arxiv.org/abs/1512.03385

'''

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['GL_ResNet', 'resnet32', 'resnet110']

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class GL_ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, num_branches = 3, input_channel=64, factor=8, en = False, zero_init_residual=False, 
        groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, KD = False):
        super(GL_ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        self.en = en
        self.num_branches = num_branches
        
        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        fix_inplanes=self.inplanes    # 32
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        for i in range(num_branches):
            setattr(self, 'layer3_' + str(i), self._make_layer(block, 64, layers[2], stride=2))
            self.inplanes = fix_inplanes  ##reuse self.inplanes
            setattr(self, 'classifier3_' +str(i), nn.Linear(64 * block.expansion, num_classes))
        
        self.query_weight = nn.Linear(input_channel, input_channel//factor, bias = False)
        self.key_weight = nn.Linear(input_channel, input_channel//factor, bias = False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                    
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
        
    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)            # B x 16 x 32 x 32

        x = self.layer1(x)          # B x 16 x 32 x 32
        x = self.layer2(x)          # B x 32 x 16 x 16
        x_3 = getattr(self,'layer3_0')(x)   # B x 64 x 8 x 8
        x_3 = self.avgpool(x_3)             # B x 64 x 1 x 1
        x_3 = x_3.view(x_3.size(0), -1)     # B x 64
        proj_q = self.query_weight(x_3)     # B x 8
        proj_q = proj_q[:, None, :]
        proj_k = self.key_weight(x_3)       # B x 8  
        proj_k = proj_k[:, None, :]
        x_3_1 = getattr(self, 'classifier3_0')(x_3)     # B x num_classes
        pro = x_3_1.unsqueeze(-1) 
        if self.en:
            for i in range(1, self.num_branches):
                temp = getattr(self, 'layer3_'+str(i))(x)
                temp = self.avgpool(temp)       # B x 64 x 1 x 1
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
                temp = self.avgpool(temp)       # B x 64 x 1 x 1
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
            temp = self.avgpool(temp)       # B x 64 x 1 x 1
            temp = temp.view(temp.size(0), -1)   
            temp_out = getattr(self, 'classifier3_' + str(self.num_branches - 1))(temp)
            return pro, x_m, temp_out
        
def resnet32(pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-32 model.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    
    model = GL_ResNet(BasicBlock, [5, 5, 5], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model

def resnet110(pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-110 model.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    
    model = GL_ResNet(Bottleneck, [12, 12, 12], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model
    