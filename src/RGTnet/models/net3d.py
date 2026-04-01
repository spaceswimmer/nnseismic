import torch.nn as nn
from . import modules3d, rgt3d

class model(nn.Module):
    def __init__(self, param):
        super(model, self).__init__()
        self.E = modules3d.E(rgt3d.encoder_backbone(param["input_channels"]))
        self.D = modules3d.D(param["encoder_channels"])   
        self.R = modules3d.R(param["decoder_channels"])

    def forward(self, x):
        x_block0, x_block1, x_block2, x_block4 = self.E(x)
        x_decoder = self.D(x_block0, x_block1, x_block2, x_block4)
        out = self.R(x_decoder)
        return out
