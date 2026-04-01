import os
import time
import json
from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from torchsummary import summary

import utils
from models import net3d
from lossf.loss import *
from lossf.metrics import *
from data.dataloader import Dataset
from data.augments import Reshape, ToTensor

def infer_model(model, dataloader):
    # Define loss function
    infer(model, dataloader, opt.only_load_input, bin_path, picture_path, device)

def infer(model, dataloader, only_load_input, bin_path, picture_path, device):
    model.eval()
    pred_sample_list = []
    if not only_load_input:
        for batch_index, (seis, rgt) in enumerate(dataloader):
            seis = seis.to(device)
            with torch.no_grad():
                rgt_pred = model(seis)
            seis = seis.squeeze().cpu().numpy()
            rgt = rgt.squeeze().numpy()
            rgt_pred = rgt_pred.squeeze().cpu().numpy()
            save(seis, rgt_pred, batch_index, bin_path)
            draw(seis, rgt, rgt_pred, batch_index, picture_path)
            pred_sample = {"seis":seis,"rgt":rgt,
                           "pred_rgt":rgt_pred}
            pred_sample_list.append(pred_sample)
    else:      
        for batch_index, (seis) in enumerate(dataloader):
            seis = seis.to(device)
            with torch.no_grad():
                rgt_pred = model(seis)
            seis = seis.squeeze().cpu().numpy()
            rgt_pred = rgt_pred.squeeze().cpu().numpy()
            save(seis, rgt_pred, batch_index, bin_path)
            draw(seis, None, rgt_pred, batch_index, picture_path)
            pred_sample = {"seis":seis, "pred_rgt":rgt_pred}
            pred_sample_list.append(pred_sample)
    return pred_sample_list

def draw(seis, rgt, rgt_pre, batch_index, picture_path, epoch=0, sec_idx= None):
    if sec_idx is None:
       sec_idx = seis.shape[-1]//2
    seis2d = seis[..., sec_idx]
    rgt_pre2d = rgt_pre[..., sec_idx]
    
    path_label = os.path.join(picture_path, "label")
    utils.makeDir(path_label)
    if rgt is not None:
      rgt2d = rgt[..., sec_idx]
      utils.draw_img(rgt2d, cmap='jet',
        path=os.path.join(path_label,f'rgt_{batch_index}_{sec_idx}.png'))
    utils.draw_img(seis2d, cmap='gray',
          path=os.path.join(path_label,f'seis_{batch_index}_{sec_idx}.png'))

    path_pred = os.path.join(picture_path, "pred")
    utils.makeDir(path_pred)
    utils.draw_img(rgt_pre2d, cmap='jet',
      path=os.path.join(path_pred,f'rgt_{batch_index}_{sec_idx}.png'))
    utils.draw_img(seis2d, ctr=rgt_pre2d, cmap='gray',
      path=os.path.join(path_pred,f'seis_{batch_index}_{sec_idx}.png'))

def save(seis, rgt_pred, batch_index, bin_path):
    seis_bin_path = os.path.join(bin_path, "seis")
    pred_bin_path = os.path.join(bin_path, "pred")
    utils.makeDir(seis_bin_path)
    utils.makeDir(pred_bin_path)
    utils.writeData3d(rgt_pred, os.path.join(pred_bin_path, str(batch_index)+'.dat'))
    utils.writeData3d(seis, os.path.join(seis_bin_path, str(batch_index)+'.dat'))    
    
def init_run():
    utils.makeDir(session_path)
    utils.makeDir(picture_path)
    utils.makeDir(bin_path)
    opt_dict = json.dumps(vars(opt), indent=4, separators=(',', ':'), ensure_ascii=False)
    f = open(os.path.join(session_path,'parameters_json.json'), 'w')
    f.write(opt_dict)
    f.close()

if __name__ == "__main__":
    
    from options.test_options import TestOptions3d
    opt = TestOptions3d().parse()
    session_name = '_'.join((opt.session_name,'Test'))

    session_path = os.path.join(opt.sessions_path, session_name)
    picture_path = os.path.join(session_path, "picture")
    bin_path = os.path.join(session_path, "bin")
    
    # Preparation
    init_run()

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Parameters
    n1, n2, n3 = opt.shape[0], opt.shape[1], opt.shape[2]
    n_channels = opt.n_channels
    batch_size = opt.batch_size
    data_dir = opt.dataroot

    data_path = os.path.join(data_dir, "seis")
    data_list = os.listdir(data_path)
    list_IDs = utils.sort_list_IDs(data_list)
    if opt.dataset_size < len(list_IDs):
      list_IDs = list_IDs[:opt.dataset_size]

    # Dataset
    dataset = Dataset(root_dir=data_dir, list_IDs=list_IDs,
                      transform=transforms.Compose([
                          Reshape((n1, n2, n3, n_channels)),
                          ToTensor(),
                      ]),
                      only_load_input=opt.only_load_input)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=opt.num_workers)
    dataloader_val = None

    # Define model
    param_model = {}
    param_model['input_channels'] = 1
    param_model['encoder_channels'] = 512      
    param_model['decoder_channels'] = 16
    model = net3d.model(param_model)

    # Load model
    if opt.trained_model is not None:
        if use_cuda: 
          model.load_state_dict(torch.load(opt.trained_model))
        else:
          model.load_state_dict(torch.load(opt.trained_model, map_location='cpu'))

    params = list(model.named_parameters())

    # Send model to GPU
    if use_cuda:
        num_GPU = torch.cuda.device_count()
        model = torch.nn.DataParallel(model, device_ids=range(num_GPU)).to(device)
    else:
        print(f"CPU mode")
        model = model.to(device)

    # Train model
    infer_model(model, dataloader)
