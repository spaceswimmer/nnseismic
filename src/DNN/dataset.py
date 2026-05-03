import os
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms


def mea_std_norm(x):
    x = (x - np.mean(x)) / np.std(x)
    return x


class Reshape(object):
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, data):
        data = np.reshape(data, self.output_size)
        return data


class ToTensor(object):
    def __call__(self, data):
        data = data.transpose((3, 2, 1, 0))
        return data


def sort_list_IDs(list_IDs):
    list_nums = [int(i.split(".")[0]) for i in list_IDs]
    list_sort = sorted(enumerate(list_nums), key=lambda x: x[1])
    list_index = [i[0] for i in list_sort]
    list_IDs_new = [list_IDs[i] for i in list_index]
    return list_IDs_new


class SeismicDataset(data.Dataset):
    def __init__(
        self, root_dir, list_IDs, shape=(128, 128, 128, 1), only_load_input=False
    ):
        self.list_IDs = list_IDs
        self.root_dir = root_dir
        self.only_load_input = only_load_input
        self.seis_dir = os.path.join(root_dir, "seis")
        self.shape = shape
        self.transform = transforms.Compose(
            [
                Reshape(shape),
                ToTensor(),
            ]
        )
        if not self.only_load_input:
            self.rgt_dir = os.path.join(root_dir, "rgt")

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        seis_path = os.path.join(self.seis_dir, ID)

        X = np.fromfile(seis_path, dtype=np.float32)
        X = self.transform(X)
        X = mea_std_norm(X)
        X = torch.from_numpy(X).bfloat16()

        if not self.only_load_input:
            rgt_path = os.path.join(self.rgt_dir, ID)
            Y = np.fromfile(rgt_path, dtype=np.float32)
            Y = self.transform(Y)
            Y = mea_std_norm(Y)
            Y = torch.from_numpy(Y).bfloat16()
            return X, Y
        else:
            return X


def create_dataloader(
    dataroot,
    shape=(128, 128, 128, 1),
    batch_size=1,
    dataset_size=float("inf"),
    num_workers=0,
    shuffle=True,
):
    data_path = os.path.join(dataroot, "seis")
    if not os.path.exists(data_path):
        raise ValueError(f"Data path {data_path} does not exist")

    data_list = os.listdir(data_path)
    list_IDs = sort_list_IDs(data_list)

    if dataset_size < len(list_IDs):
        list_IDs = list_IDs[: int(dataset_size)]

    dataset = SeismicDataset(root_dir=dataroot, list_IDs=list_IDs, shape=shape)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return dataloader
