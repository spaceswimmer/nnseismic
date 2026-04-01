import torch
import math
import numpy as np
import torch.nn as nn

from .ssim3d import MultiScaleSSIMLoss3d as ms_ssim3d
SSIM = ms_ssim3d()

def MRPD(output, target):
    a = (output-target).abs()  
    b = output.abs()+target.abs()
    return (a/b).mean()*2

class Result(object):
    def __init__(self):
        self.mse, self.rmse, self.mae = 0, 0, 0
        self.absrel = 0
        self.mrpd = 0
        self.ssim = 0

    def set_to_worst(self):
        self.mse, self.rmse = np.inf, np.inf
        self.mae, self.absrel = np.inf, np.inf
        self.mrpd, self.ssim = np.inf, np.inf

    def update(self, mse, rmse, mae, absrel, mrpd, ssim):
        self.mse, self.mae = mse, mae
        self.absrel, self.rmse = absrel, rmse
        self.mrpd, self.ssim = mrpd, ssim

    def evaluate(self, output, target):
        abs_diff = (output - target).abs()
        self.mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())
        self.absrel = float((abs_diff / target).mean())
        self.mrpd = float(MRPD(output, target)) 
        self.ssim = float(1 - SSIM(output, target)) 

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0
        self.sum_mse, self.sum_rmse = 0, 0
        self.sum_mae, self.sum_absrel = 0, 0
        self.sum_mrpd, self.sum_ssim = 0, 0

    def update(self, result, n=1):
        self.count += n
        self.sum_mse += n*result.mse
        self.sum_rmse += n*result.rmse
        self.sum_mae += n*result.mae
        self.sum_absrel += n*result.absrel
        self.sum_mrpd += n*result.mrpd
        self.sum_ssim += n*result.ssim

    def average(self):
        avg = Result()
        avg.update(
            self.sum_mse / self.count, self.sum_rmse / self.count, 
            self.sum_mae / self.count, self.sum_absrel / self.count,
            self.sum_mrpd / self.count, self.sum_ssim / self.count)
        return avg
