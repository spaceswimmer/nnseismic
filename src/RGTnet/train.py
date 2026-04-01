import os
import time
import json
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torchsummary import summary
from torch.utils.data import DataLoader

from tqdm import tqdm

import utils
from models import net3d
from lossf.loss import *
from lossf.metrics import *
from data.dataloader import Dataset
from data.augments import Reshape, ToTensor
from data.augments import HorizontalFlip1, HorizontalFlip2, VerticalFlip
from options.train_options import TrainOptions3d

opt = TrainOptions3d().parse()
session_name = '_'.join((opt.session_name,'Train'))
session_path = os.path.join(opt.sessions_path, session_name)
picture_path = os.path.join(session_path, "picture")
checkpoint_path = os.path.join(session_path, "checkpoint")
history_path = os.path.join(session_path, "history")

def train_model(model, optimizer, dataloader, scheduler, num_epochs, opt, dataloader_val = None):

    # Define loss function
    if opt.loss_type == 'MSE':
        criterion = mse3DLoss()
    elif opt.loss_type == "SSIM":
        criterion = ssim3DLoss()

    train_loss_history, test_loss_history = [], []
    for epoch in range(num_epochs):

        train_epoch_loss_ifo = train(model, dataloader, optimizer, criterion, scheduler, epoch, opt)
        train_loss_history.append(train_epoch_loss_ifo)

        if opt.valid and epoch % opt.valid_interval == 0:
          test_epoch_loss_ifo = valid(model, dataloader_val, criterion, epoch, opt)
          test_loss_history.append(test_epoch_loss_ifo)
        
        # Save checkpoint
        if epoch % opt.checkpoint_interval == 0:
          save_name = os.path.join(checkpoint_path, "{}.pth".format(epoch))
          torch.save(model.module.state_dict(), save_name)

        if epoch % opt.history_interval == 0:
          utils.save_training_history(train_loss_history, os.path.join(history_path, f"train_history.txt"))
          if opt.valid:
            utils.save_training_history(test_loss_history, os.path.join(history_path, f"test_history.txt"))

    return model

def train(model, dataloader, optimizer, criterion, scheduler, epoch, opt):

    model.train()
    running_loss = 0.0
    average_meter = AverageMeter()

    for i, param_group in enumerate(optimizer.param_groups):
        current_lr = float(param_group['lr'])

    for batch_index, (seis, rgt) in enumerate(dataloader):

        if opt.data_augmentation:
            seis = torch.cat((seis, HorizontalFlip1(seis)), dim=0)
            rgt = torch.cat((rgt, HorizontalFlip1(rgt)), dim=0)

        seis, rgt = seis.to(device), rgt.to(device)

        with torch.autograd.detect_anomaly():
            rgt_pred = model(seis)
            loss = criterion(rgt_pred, rgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss += float(loss)
        result= Result()
        result.evaluate(rgt.data, rgt_pred.data)
        average_meter.update(result, seis.size(0))

    epoch_loss = running_loss / len(dataloader)
    tqdm.write('=> Train Epoch:{0}'
          ' Loss={Loss:.5f}'
          '\nLearning rate: {1}'
          '\n-------------------------------------------------------------------------------------------'.format(
          epoch, current_lr, Loss=epoch_loss))
    epoch_loss_ifo = {'EPOCH': epoch,
    criterion.getLossName(): epoch_loss,
    'LR': current_lr}
    scheduler.step(epoch_loss)
    return epoch_loss_ifo

def valid(model, dataloader, criterion, epoch, opt):

    model.eval()
    running_loss = 0.0
    average_meter = AverageMeter()

    for batch_index, (seis, rgt) in enumerate(dataloader):

        seis, rgt = seis.to(device), rgt.to(device)

        with torch.no_grad():
            rgt_pred = model(seis)
            loss = criterion(rgt_pred, rgt)
       
        running_loss += loss.data
        result= Result()
        result.evaluate(rgt.data, rgt_pred.data)
        average_meter.update(result, seis.size(0))

        seis = seis.squeeze().cpu().numpy()
        rgt = rgt.squeeze().cpu().numpy()
        rgt_pred = rgt_pred.squeeze().cpu().numpy()
        draw(seis, rgt, rgt_pred, batch_index, epoch)

    epoch_loss = running_loss / len(dataloader)
    avg = average_meter.average()
    tqdm.write('=> Valid Epoch:{0}'
          ' Loss={Loss:.5f}'.format(
          epoch, Loss=epoch_loss))
    tqdm.write('=> RGT:'
          ' MSE={result.mse:.3f}({average.mse:.3f})'
          ' RMSE={result.rmse:.3f}({average.rmse:.3f})'
          ' MAE={result.mae:.3f}({average.mae:.3f})'
	      ' MRPD={result.mrpd:.3f}({average.mrpd:.3f})'
	      ' SSIM={result.ssim:.3f}({average.ssim:.3f})'
          '\n==========================================================================================='.format(
        result=result, average=avg))

    epoch_loss_ifo = {'EPOCH': epoch,
    'MAE': avg.mae,
    'MSE': avg.mse,
    'RMSE': avg.rmse,
    'MRPD': avg.mrpd,
    'SSIM': avg.ssim
    }
    return epoch_loss_ifo

def draw(seis, rgt, rgt_pre, batch_index, epoch=0, sec_idx=None):
    if sec_idx is None:
      sec_idx = seis.shape[-1]//2 
    seis2d = seis[..., sec_idx]
    rgt2d = rgt[..., sec_idx]
    rgt_pre2d = rgt_pre[..., sec_idx]
    path_pred_epoch = os.path.join(picture_path, "pred", f"epoch_{epoch}")
    utils.makeDir(path_pred_epoch)
    utils.draw_img(rgt_pre2d, cmap='jet',
      path=os.path.join(path_pred_epoch,f'rgt_{epoch}_{batch_index}_{sec_idx}.png'))
    utils.draw_img(seis2d, ctr=rgt_pre2d, cmap='gray', 
      path=os.path.join(path_pred_epoch,f'seis_{epoch}_{batch_index}_{sec_idx}.png'))      
    if epoch == 0:
        path_label_epoch = os.path.join(picture_path, "label")
        utils.makeDir(path_label_epoch)
        utils.draw_img(rgt2d, cmap='jet',
          path=os.path.join(path_label_epoch,f'rgt_{epoch}_{batch_index}_{sec_idx}.png'))
        utils.draw_img(seis2d, ctr=rgt2d, cmap='gray', 
          path=os.path.join(path_label_epoch,f'seis_{epoch}_{batch_index}_{sec_idx}.png'))      

def init_run():
    utils.makeDir(session_path)
    utils.makeDir(checkpoint_path)
    utils.makeDir(history_path)   
    opt_dict = json.dumps(vars(opt), indent=4, separators=(',', ':'), ensure_ascii=False)
    f = open(os.path.join(session_path,'parameters_json.json'), 'w')
    f.write(opt_dict)
    f.close()

if __name__ == "__main__":
    
    # init
    init_run()

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Parameters
    n1, n2, n3 = opt.shape[0], opt.shape[1], opt.shape[2]
    n_channels = opt.n_channels
    num_epochs = opt.nepochs
    batch_size = opt.batch_size
    data_dir = opt.dataroot
    data_dir_val = opt.dataroot_val
    
    # Get train file list
    data_path = os.path.join(data_dir, "seis")
    data_list = os.listdir(data_path)
    list_IDs = utils.sort_list_IDs(data_list)

    if opt.dataset_size < len(list_IDs):
      list_IDs = list_IDs[:opt.dataset_size]

    # Train dataset
    dataset = Dataset(root_dir=data_dir, list_IDs=list_IDs,
                      transform=transforms.Compose([
                          Reshape((n1, n2, n3, n_channels)),
                          ToTensor(),
                      ]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=opt.num_workers)
    dataloader_val = None

    # Get valid file list
    data_path = os.path.join(data_dir_val, "seis")
    data_list = os.listdir(data_path)
    list_IDs = utils.sort_list_IDs(data_list)

    if opt.dataset_size_val < len(list_IDs):
      list_IDs = list_IDs[:opt.dataset_size_val]

    # Valid dataset
    if opt.valid:
        dataset_val = Dataset(root_dir=data_dir_val, list_IDs=list_IDs,
                          transform=transforms.Compose([
                              Reshape((n1, n2, n3, n_channels)),
                              ToTensor(),
                          ]))
        dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False,
                                num_workers=opt.num_workers)

    # Define model
    param_model = {}
    param_model['input_channels'] = 1
    param_model['encoder_channels'] = 512
    param_model['decoder_channels'] = 16
    model = net3d.model(param_model)
 
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=opt.lr_patience, factor=opt.lr_factor)

    # Load model
    if opt.pretrained_model is not None:
        if use_cuda: 
          model.load_state_dict(torch.load(opt.pretrained_model))
        else:
          model.load_state_dict(torch.load(opt.pretrained_model, map_location='cpu'))

    # Send model to GPU
    num_GPU = torch.cuda.device_count()
    model = torch.nn.DataParallel(model, device_ids=range(num_GPU)).to(device)

    # Train model
    model = train_model(model, optimizer, dataloader, scheduler, num_epochs, opt, dataloader_val=dataloader_val)
