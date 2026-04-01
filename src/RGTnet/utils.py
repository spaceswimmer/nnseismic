import os
import struct
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt  
plt.switch_backend('agg')

def sort_list_IDs(list_IDs):
    list_nums = [int(i.split(".")[0]) for i in list_IDs]
    list_sort = sorted(enumerate(list_nums), key=lambda x:x[1])
    list_index = [i[0] for i in list_sort]
    list_IDs_new = [list_IDs[i] for i in list_index]
    return list_IDs_new

def readData3d(n1,n2,n3,path):
    dat = np.fromfile(path, dtype=np.float32)
    dat = dat.reshape((n1,n2,n3,1)).transpose((3,2,1,0))
    return dat

def makeDir(path):
    if not path.endswith('/'):
        path = path + '/'
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True) 

def draw_img(img, path=None, ctr=None, msk=None, cmap="jet", method="bilinear"):
    plt.imshow(img,cmap=cmap, interpolation=method)
    if msk is not None:
        plt.imshow(msk, alpha=0.4, cmap='jet', interpolation='nearest')
    if ctr is not None:
        plt.contour(ctr,np.linspace(np.min(ctr),np.max(ctr),30),colors='black',linewidths=2)
    plt.colorbar(fraction=0.023,pad=0.02)
    if path is not None:
        plt.savefig(path)
    plt.close()

def AdjustLearningRate(optimizer, epoch, init_lr):
    """Adjust learning rate"""
    lr = init_lr * (0.1 ** ((epoch + 1) // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def min_max_norm(x):
    x = x - np.min(x)
    x = x / np.max(x)
    return x

def mea_std_norm(x):
    x = (x - np.mean(x)) / np.std(x)
    return x

def save_training_history(history, path):
    with open(path, "w") as f:
        n = len(history[0])
        keys = list(history[0].keys())
        for i in range(n-1):
            f.write(f"{keys[i]}\t")
        f.write(f"{keys[-1]}\n")        
        for epoch_history in history:
            for i in range(n-1):
                f.write(f"{epoch_history.get(keys[i])} ")
            f.write(f"{epoch_history.get(keys[-1])}\n")

def writeData3d(data, path):
    data = np.transpose(data,[2,1,0]).astype(np.single)
    data.tofile(path)

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
