import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm
torch.manual_seed(4701)
np.random.seed(4701)

# code taken & adapted from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

# Variables all proccesses need access to
# train_dir = '../../dataset/train/'
# validation_dir = '../../dataset/validation/'
load_mode = True
data_dir = '../../dataset/'
model_name = 'resnet'
num_classes = 10
batch_size = 8
num_epochs = 5
input_size = 224 # 224 x 224 images expected for resnet

def train_model(model, dataloaders, criterion, optimizer, num_epochs=15):
    """
    Returns a trained model. 
    
    Trains the input model using the provided data, criterion, and optimizer.

    Parameter model: model to be trained

    Parameter dataloaders: dictionary of dataloaders for training and validation
    Precondition: dataloders bound to keys 'train' and 'val'

    Parameter criterion: criterion for evaluating loss

    Parameter optimizer: optimizer to be used when training

    Parameter num_epochs: number of epochs to train

    """
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward pass (if training)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def histogram(counts, n = 10):
    """
    Returns a histogram of the input list of counts

    Histogram takes in a list with values in the range 0,...,n and returns a list 
    where the ith element is the number of times i appeared in counts
    """
    hist = [0]*n
    for d in counts:
        hist[d] += 1
    return hist

def eval(model, dataloader):
    running_corrects = 0
    incorrects, corrects = [], []
    for data, labels in tqdm(dataloader):
        data = data.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(data)
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)
            incorrects += [x.item() for x in preds[preds != labels.data]]
            corrects += [x.item() for x in preds[preds == labels.data]]
    # print(incorrects)
    return running_corrects.double() / len(dataloader.dataset), corrects, incorrects

def freeze(model):
    """
    Freezes every layer in the model.

    By freezing each layer, no updates are made to the weights of the model
    when training.
    """
    for param in model.parameters():
        param.requires_grad = False

def initialize_model(num_classes, pretrain=True):
    """
    Returns an initialized resnet=18 model.

    Initializes a resnet-18 model. If pretrained, then we
    use the pretrained weights and freeze the model. Otherwise,
    a fresh model with resnet-18 architecture is used. A linear
    layer is appended so the output is num_classes.
    """
    model = models.resnet18(pretrained=pretrain)
    if pretrain:
        freeze(model)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


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

        batch_mean = torch.mean(img, (0,2,3))
        batch_std = torch.std(img, (0,2,3))

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
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), \
        transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, \
        batch_size=batch_size, shuffle=True, num_workers=4)

    mean, std = calc_mean_std(train_dataloader)

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])


if __name__ == '__main__':
    print('GPU Available: ', torch.cuda.is_available())
    model_ft = initialize_model(num_classes)
    transforms = preprocess_data()

    print("Initializing Datasets and Dataloaders...")
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x if x == 'train' else 'validation'), transforms) for x in ['train', 'val']}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

    classes = image_datasets['train'].classes

    # Determine device (GPU if available, else CPU) & send model to it
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    # params_to_update = model_ft.parameters()
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    # Setup the loss fxn    
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    if not load_mode:
        model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)
        torch.save(model_ft, 'model')
        torch.save(hist, 'histogram')
    else:
        model_ft = torch.load('model')
        model_ft.to(device)
        hist = torch.load('histogram')
    
    val_acc, corrects, incorrects = eval(model_ft, dataloaders_dict['val'])
    print('Validation accuracy {:.2f}%'.format(val_acc*100))
    hist_corrects = histogram(corrects)
    hist_incorrects = histogram(incorrects)
    total_correct = sum(hist_corrects)
    total_incorrect = sum(hist_incorrects)
    print([(i, x) for i, x in enumerate(classes)])
    print([int(100*x/(total_correct+total_incorrect)) for x in hist_corrects])
    print([int(100*x/(total_correct+total_incorrect)) for x in hist_incorrects])

    train_acc, corrects, incorrects = eval(model_ft, dataloaders_dict['train'])
    print('Train accuracy {:.2f}%'.format(train_acc*100))
    hist_corrects = histogram(corrects)
    hist_incorrects = histogram(incorrects)
    total_correct = sum(hist_corrects)
    total_incorrect = sum(hist_incorrects)
    print([int(100*x/(total_correct+total_incorrect)) for x in hist_corrects])
    print([int(100*x/(total_correct+total_incorrect)) for x in hist_incorrects])
    