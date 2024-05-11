##Import Data
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class LengthEstimator(Dataset):
    """Dataset for trajectory optimization.
        shape[1] = [:,:15] = inputs; [:,15:] = targets"""

    def __init__(self, root_dir):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = load_datas(root_dir)
        self.mean = self.data[:, :7].mean(axis=0)
        self.std = self.data[:, :7].std(axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx,:]
        return sample


def generate_and_save_data(filename):
    # Generate 3 pairs of points with x, y positions
    num_points = 1000

    data = []

    # Generate random points
    points = np.random.uniform(low=0, high=5, size=(num_points, 6))

    # Randomly select start, end points
    start = points[:,:2]
    end = points[:,4:]
    # Calculate midpoints
    midpoints = points[:,2:4]

    # Calculate lengths
    lengths = np.array([np.linalg.norm(midpoints - start, axis=1) + np.linalg.norm(end - midpoints, axis=1)]).T
    print(lengths.shape)
    # Combine data
    point_data = np.hstack((points, lengths))

    # Save data to a text file
    np.savetxt(filename, point_data, fmt='%1.8f', header='X Y Length', comments='')

# Generate and save data to a text file
generate_and_save_data("005.txt")



def load_single_example(filename):
    """
    """

    with open(filename, "r") as infile:
        #print(infile)
        lines = infile.readlines()

    points = np.array([_.strip().split() for _ in lines[1:]], dtype=float) #0th line is skipped
    """points[:, :2] += np.random.random_integers(-5, 5, size=(32,2))
    print(points)
    start = points[0,:2]
    end = points[1, :2]
    midpoints = points[2:,:2]
    true_lengths = np.array([[np.linalg.norm(i - start) + np.linalg.norm(end - i)] for i in midpoints])
    #print(label)
    point_combs = np.array([[start[0], start[1], _[0], _[1], end[0], end[1]] for _ in midpoints])
    #print(point_combs.shape)
    data = np.hstack((point_combs, true_lengths))
    mean = np.mean(points, axis=0)
    std = np.std(points, axis=0)
    data = (data - mean)/std"""
    #data = data/np.max(data)
    # all point combinations with start and goal, labelled with min arg
    #datapoints[label] = torch.tensor(data, dtype=torch.float32)

    # true_lengths = torch.tensor(np.array([(np.linalg.norm(x-points[0])+np.linalg.norm(points[1]-x))
    #                         for x in points[2:]], dtype=float), dtype=torch.float32).reshape(-1,1)

    #print(true_lengths)

    return points


#load datas
def load_datas(data_folder):
    """returns a list of dictionaries from data in .txt files in given folder directory,
    with added key 'label', taken value of the point that would result with the least length.
    param: data_folder
    """
    files = [f for f in listdir(data_folder) if isfile(join(data_folder, f))]
    unclipped = []
    all_datas = []
    idx = 0
    length = 10000
    for file_name in files:
        file_name = os.path.join(data_folder, file_name)
        data = load_single_example(file_name)
        length = min(len(data), length)
        unclipped.append(data)

    data_size = len(unclipped) * length
    all_datas = np.empty(shape=(data_size, 7), dtype=np.float64)
    for data in unclipped:
        all_datas[idx:idx + length] = data[:length]
        idx += length

    print("tamamdır") if idx == data_size else print(idx, data_size)

    return np.array(all_datas)