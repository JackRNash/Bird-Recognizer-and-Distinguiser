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
from flask import Flask, jsonify, request, render_template

from sklearn.metrics import confusion_matrix
from torchvision import datasets, models, transforms
from tqdm import tqdm

net_dir = './lenet5.pth'
app = Flask(__name__)


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


@app.route('/getmethod/<jsdata>')
def get_javascript_data(jsdata):
    return jsdata


@app.route('/postmethod', methods=['POST'])
def get_post_javascript_data():
    jsdata = request.form['javascript_data']
    return jsdata


if __name__ == '__main__':
    net = Net()
    net.load_state_dict(torch.load(net_dir))
