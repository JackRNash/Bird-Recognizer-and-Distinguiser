import torch
import torch.nn as nn
import torch.nn.functional as Activation
import torch.optim as optim
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import os
import copy

from torchvision import datasets, models, transforms
from tqdm import tqdm

torch.manual_seed(4701)
np.random.seed(4701)

data_dir = '../../dataset/'
net_dir = './lenet5.pth'
batch_size = 8
num_epochs = 10


class Net(nn.Module):
    """
    This class represents a basic model of the Lenet-5 CNN modified to operate on
    256 x 256 images.

    CITATION: This class and its methods were taken and adapted from the following
        source: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(61 * 61 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, val):
        """
        Returns the prediction corresponding to a given input image.

        Parameter val: the image to predict
        Precondition: val is 256x256x3 Torch Tensor object
        """
        val = self.pool1(Activation.relu(self.conv1(val)))
        val = self.pool2(Activation.relu(self.conv2(val)))
        val = val.view(-1, 61 * 61 * 16)
        val = Activation.relu(self.fc1(val))
        val = Activation.relu(self.fc2(val))
        val = self.fc3(val)
        return val


def calc_mean_std(dataloader):
    """
    Returns a list of the average RGB values and a list of the standard of
    the RGB values.

    Extracts the mean and standard of RGB values of every image in dataloader
    and then returns the average.

    Parameter dataloader: images to find the mean and standard of
    Precondition: dataloader is a Torch dataloader object

    CITATION: This function was taken and adapted from the following source:
        https://forums.fast.ai/t/image-normalization-in-pytorch/7534/7
    """
    mean = []
    std = []

    # Find and append the mean and standard of each image in dataloader
    for data in dataloader:
        img, _ = data

        batch_mean = torch.mean(img, (0, 2, 3))
        batch_std = torch.std(img, (0, 2, 3))

        mean.append(batch_mean)
        std.append(batch_std)

    # Find and return the mean and standard of every image in dataloader
    mean = np.mean([m.numpy() for m in mean], axis=0)
    std = np.mean([s.numpy() for s in std], axis=0)

    return mean, std


def preprocess_data():
    """
    Returns the transformation necessary to normalize the image data.

    Creates a normalization transformation that results in the training data
    having a mean of 0 and a standard deviation of 1.
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Creates a dataloader object for the training and validation sets
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                         transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size, shuffle=True, num_workers=4)

    mean, std = calc_mean_std(train_dataloader)

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def train_model(dataloader, opt):
    """
    Trains the CNN on the training data.

    Parameter dataloader: images used to train the CNN
    Precondition: dataloader is a Torch dataloader object

    Parameter opt: optimizer used to update the CNN after each pass
    Precondition: opt is a Torch optim object

    CITATION: This function was taken and adapted from the following source:
        https://forums.fast.ai/t/image-normalization-in-pytorch/7534/7
    """
    for _ in range(num_epochs):
        for data in dataloader:
            imgs, labels = data

            # Zero the parameter gradients
            opt.zero_grad()

            # Perform forward-backward pass, then update optimizer
            outputs = net(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()


def test_model(net, dataloader):
    """
    Calculates the accuracy of the CNN on an unseen dataset.

    Parameter net: the CNN trained on the training data, used for bird
    classification
    Precondition: net is a Net object

    Parameter dataloader: images used to validate the CNN
    Precondition: dataloader is a Torch dataloader object

    CITATION: This function was taken and adapted from the following source:
        https://forums.fast.ai/t/image-normalization-in-pytorch/7534/7
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            imgs, labels = data
            outputs = net(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            for i in range(len(predicted)):
                if predicted[i] == labels[i]:
                    correct += 1

    print('Accuracy of the network on the ' + str(total) +
          ' test images: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    # Obtain necessary transformation code
    data_transform = preprocess_data()

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir,
                                                           x if x == 'train' else 'validation'), data_transform)
                      for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                       batch_size=batch_size, shuffle=True, num_workers=0)
                        for x in ['train', 'val']}

    # Create model
    net = Net()

    # Create loss function & establish optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train_model(dataloaders_dict['train'], optimizer)

    """ Remove this store/load functionality eventually """
    # Store model
    torch.save(net.state_dict(), net_dir)

    # Load model
    net = Net()
    net.load_state_dict(torch.load(net_dir))

    test_model(net, dataloaders_dict['val'])
