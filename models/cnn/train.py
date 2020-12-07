import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os

from torchvision import datasets, models, transforms

data_dir = '../../dataset/'
batch_size = 1

def calc_mean_std(dataloader):
    """
    TO DO

    CITATION: This function was taken and adapted from the following source:
        https://forums.fast.ai/t/image-normalization-in-pytorch/7534/7
    """
    mean = []
    std = []

    # Find and append the mean and standard of each image in dataloader
    for i, data in enumerate(dataloader, 0):
        img, label = data
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
    # Creates a dataloader object for the training and validation sets
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), None)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    mean, std = calc_mean_std(train_dataloader)

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])


if __name__ == '__main__':
    transformation = preprocess_data()
