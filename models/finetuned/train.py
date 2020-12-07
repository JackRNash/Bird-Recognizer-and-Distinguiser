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
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
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
    # takes in a list containing values in the range 0,...,n and returns
    # a list hist where hist[i] is the number of times i appeared in counts
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
    for param in model.parameters():
        param.requires_grad = False

def initialize_model(num_classes, pretrain=True):
    model = models.resnet18(pretrained=pretrain)
    if pretrain:
        freeze(model)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def calc_mean_std(dataloader):
    # from https://forums.fast.ai/t/image-normalization-in-pytorch/7534/7
    pop_mean = []
    pop_std = []
    for data in dataloader:
        # shape (batch_size, 3, height, width)
        img, label = data
        numpy_image = img.numpy()
        
        # shape (3,)
        batch_mean = np.mean(numpy_image, axis=(0,2,3))
        batch_std = np.std(numpy_image, axis=(0,2,3))
        
        pop_mean.append(batch_mean)
        pop_std.append(batch_std)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    pop_mean = np.array(pop_mean).mean(axis=0)
    pop_std = np.array(pop_std).mean(axis=0)
    return pop_mean, pop_std


if __name__ == '__main__':
    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)
    print('GPU Available: ', torch.cuda.is_available())
    model_ft = initialize_model(num_classes)

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.48150042, 0.49749625, 0.4631295], [0.22679698, 0.22562513, 0.26397094])
            # hard coded mean and std respectively calculated using calc_mean_std
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.48150042, 0.49749625, 0.4631295], [0.22679698, 0.22562513, 0.26397094])
            # hard coded mean and std respectively calculated using calc_mean_std
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x if x == 'train' else 'validation'), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

    classes = image_datasets['train'].classes
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)

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
    