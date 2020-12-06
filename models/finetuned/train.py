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
data_dir = '../../dataset/'
model_name = 'resnet'
num_classes = 10
batch_size = 8
num_epochs = 1
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
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
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
    for i, data in enumerate(dataloader, 0):
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
            transforms.Normalize([0.4773241, 0.5002451, 0.4575769], [0.22043052, 0.21716149, 0.2665576])
            # hard coded mean and std respectively calculated using calc_mean_std
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x if x == 'train' else 'validation'), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

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
    print("Params to learn:")
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
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

    # Initialize the non-pretrained version of the model used for this run
    # scratch_model,_ = initialize_model(num_classes)
    # scratch_model = scratch_model.to(device)
    # scratch_optimizer = optim.SGD(scratch_model.parameters(), lr=0.001, momentum=0.9)
    # scratch_criterion = nn.CrossEntropyLoss()
    # _,scratch_hist = train_model(scratch_model, dataloaders_dict, scratch_criterion, scratch_optimizer, num_epochs=num_epochs)

    # # Plot the training curves of validation accuracy vs. number
    # #  of training epochs for the transfer learning method and
    # #  the model trained from scratch
    # ohist = []
    # shist = []

    # ohist = [h.cpu().numpy() for h in hist]
    # shist = [h.cpu().numpy() for h in scratch_hist]

    # plt.title("Validation Accuracy vs. Number of Training Epochs")
    # plt.xlabel("Training Epochs")
    # plt.ylabel("Validation Accuracy")
    # plt.plot(range(1,num_epochs+1),ohist,label="Pretrained")
    # plt.plot(range(1,num_epochs+1),shist,label="Scratch")
    # plt.ylim((0,1.))
    # plt.xticks(np.arange(1, num_epochs+1, 1.0))
    # plt.legend()
    # plt.show()