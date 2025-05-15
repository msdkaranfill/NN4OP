##Import Data
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TrajectoryOptimizationDataset(Dataset):
    """Dataset for trajectory optimization.
        shape[1] = [:,:15] = inputs; [:,15:] = targets"""

    def __init__(self, root_dir: object) -> object:
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = load_datas(root_dir)
        self.mean = self.data[:, :15].mean(axis=0)
        self.std = self.data[:, :15].std(axis=0)
        
        # Handle near-zero velocities in the true velocities only
        eps = 1e-3
        # Clean target velocities (indices 15:18)
        self.data[:, 15:18] = np.where(np.abs(self.data[:, 15:18]) < eps, 0, self.data[:, 15:18])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx,:]
        return sample




def load_datas(data_folder):
    """returns an np array  consist of all data in the folder.
    shape:[# of files x # of samples, 19]
    param: data_folder
    """
    files = [f for f in listdir(data_folder) if isfile(join(data_folder, f))]
    unclipped = []
    all_datas = []
    idx = 0
    length = 500000
    for file_name in files:
        file_name = os.path.join(data_folder, file_name)
        data = load_single_example(file_name)
        length = min(len(data), length)
        unclipped.append(data)

    data_size = len(unclipped)*length
    all_datas = np.empty(shape=(data_size, 19), dtype=np.float64)
    for data in unclipped:
        all_datas[idx:idx+length] = data[:length]
        idx += length

    print(all_datas.shape[0] if idx == data_size else print(idx, data_size))

    return np.array(all_datas)



def load_single_example(filename):
    """
    """
    print(filename)
    with open(filename, "r") as infile:
        lines = infile.readlines()
        points = np.array([_.strip().split() for _ in lines], dtype=float)

    return points



