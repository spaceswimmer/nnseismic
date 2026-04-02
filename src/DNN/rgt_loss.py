import torch
import torch.nn as nn
from src.RGTnet.lossf.ssim3d import MultiScaleSSIMLoss3d

class SSIM3DLoss(nn.Module):
    """
    Direct implementation from the paper.
    MS-SSIM loss for 3D data.
    """
    def __init__(self, channel=1):
        super().__init__()
        self.ssim = MultiScaleSSIMLoss3d(channel=channel)
    
    def forward(self, output, target):
        # Paper uses (1 - SSIM) as loss
        return 1 - self.ssim(output, target)


class CombinedLoss(nn.Module):
    """
    Simple combination of MSE + SSIM (as many papers do).
    """
    def __init__(self, lambda_ssim=0.5, lambda_mse=0.5, channel=1):
        super().__init__()
        self.lambda_ssim = lambda_ssim
        self.lambda_mse = lambda_mse
        self.ssim_loss = SSIM3DLoss(channel=channel)
        self.mse_loss = nn.MSELoss()
    
    def forward(self, output, target):
        ssim_loss = self.ssim_loss(output, target)
        mse_loss = self.mse_loss(output, target)
        
        total_loss = self.lambda_ssim * ssim_loss + self.lambda_mse * mse_loss
        
        return total_loss, {'ssim': ssim_loss.item(), 'mse': mse_loss.item()}