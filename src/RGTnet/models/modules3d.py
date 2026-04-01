import torch
import torch.nn.functional as F
import torch.nn as nn

def conv5x5(in_planes, out_planes, stride=1):
    "5x5 convolution with padding"
    return nn.Conv3d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution without padding"
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)

class Attention_block(nn.Module):
    def __init__(self, F_g, F_x, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_x, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class SELayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class UP(nn.Sequential):
    def __init__(self, num_input_features_g, num_input_features_x, num_output_features, at_block=True, se_block=True):
        super(UP, self).__init__()
        # attention block
        if at_block:
          self.atte = Attention_block(num_input_features_g, num_input_features_x, num_input_features_x)
        else:
          self.atte = None
        num_input_features = num_input_features_g + num_input_features_x

        # conventional conv
        self.conv = conv5x5(num_input_features, num_output_features)
        self.bn = nn.InstanceNorm3d(num_output_features)
        self.relu = nn.ReLU(inplace=True)            

        # se blovk
        if se_block:
          self.se = SELayer(num_output_features)
        else:
          self.se = None

    def forward(self, x, x1, size):
        x = F.interpolate(x, size=size, mode='trilinear', align_corners=True)
        if self.atte is not None:
          x1 = self.atte(x,x1)
        x = torch.cat([x,x1], dim=1)
        out = self.relu(self.bn(self.conv(x)))
        if self.se is not None:
          out = self.se(out)
        return out

class UPSE(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(UPSE, self).__init__()
        self.conv = conv5x5(num_input_features, num_output_features*2)
        self.bn = nn.InstanceNorm3d(num_output_features*2)
        self.relu = nn.ReLU(inplace=True)            
        self.se = SELayer(num_output_features*2, num_output_features)

    def forward(self, x, x1, size):
        x = F.interpolate(x, size=size, mode='trilinear', align_corners=True) 
        x = torch.cat((x,x1), dim=1)
        out = self.relu(self.bn(self.conv(x)))
        out = self.se(out)
        return out

class R(nn.Module):
    def __init__(self, num_features):
        super(R, self).__init__()
        self.conv0 = nn.Conv3d(num_features, 1, kernel_size=1,
                               stride=1, padding=0, bias=True)
        self.bn0 = nn.InstanceNorm3d(1)
        self.identity = nn.Identity()

    def forward(self, x):        
        x1 = self.identity(self.bn0(self.conv0(x)))
        return x1

class E(nn.Module):
    def __init__(self, original_model):
        super(E, self).__init__()

        self.layer0 = original_model.layer0
        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4
        self.layer5 = original_model.layer5
        self.layer7 = original_model.layer7
        self.layer8 = original_model.layer8

    def forward(self, x):

        x_block0 = self.layer0(x)      

        x = self.layer1(x_block0)        
        x_block1 = self.layer2(x)        
        x_block2 = self.layer3(x_block1)        
        x_block3 = self.layer4(x_block2)      

        x = self.layer5(x_block3)        
        x = self.layer7(x)        
        x_block4 = self.layer8(x)        

        return x_block0, x_block1, x_block2, x_block4

class D(nn.Module):
    def __init__(self, num_features = 512):
        super(D, self).__init__()
        self.conv = nn.Conv3d(num_features, num_features,
                              kernel_size=1, stride=1, bias=False)
        self.bn = nn.InstanceNorm3d(num_features)

        self.up3 = UP(512, 64, num_output_features=256)
        self.up4 = UP(256, 32, num_output_features=128)
        self.up5 = UP(128, 16, num_output_features=16)

    def forward(self, x_block0, x_block1, x_block2, x_block4):

        x_d0 = F.relu(self.bn(self.conv(x_block4)))
        x_d3 = self.up3(x_d0, x_block2,
                        [x_block2.size(2), x_block2.size(3), x_block2.size(4)])
        x_d4 = self.up4(x_d3, x_block1,
                        [x_block1.size(2), x_block1.size(3), x_block1.size(4)])    
        x_d5 = self.up5(x_d4, x_block0,
                        [x_block0.size(2), x_block0.size(3), x_block0.size(4)])  
        return x_d5
