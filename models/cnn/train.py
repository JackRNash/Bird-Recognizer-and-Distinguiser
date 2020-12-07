import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import os
import copy

from torchvision import datasets, models, transforms
from tqdm import tqdm

data_dir = '../../dataset/'
batch_size = 8

def calc_mean_std(dataloader):
    """
    TO DO

    CITATION: This function was taken and adapted from the following source:
        https://forums.fast.ai/t/image-normalization-in-pytorch/7534/7
    """
    mean = []
    std = []

    print("calculating mean\n")

    # Find and append the mean and standard of each image in dataloader
    for i, data in enumerate(dataloader, 0):
        print(i)
        img, label = data
        print(img)
        numpy_image = img.numpy()
        batch_mean = np.mean(numpy_image, axis=(0,2,3))
        batch_std = np.std(numpy_image, axis=(0,2,3))
        mean.append(batch_mean)
        std.append(batch_std)

    # Find and return the mean and standard of every image in dataloader
    mean = np.array(mean).mean(axis=0)
    std = np.array(std).mean(axis=0)

    print(mean)
    print(std)
    return mean, std


def preprocess_data():
    """
    TO DO
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    print("in preprocess\n")
    # Creates a dataloader object for the training and validation sets
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    print("made dataloaders\n")

    mean, std = calc_mean_std(tqdm(train_dataloader))

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])


if __name__ == '__main__':
    transformation = preprocess_data()
