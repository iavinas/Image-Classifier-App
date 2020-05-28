import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from PIL import Image


def is_gup():
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')
    return use_cuda

def load_data(data_dir='flowers', train_dir='flowers/train',valid_dir='flowers/valid',\
              test_dir='flowers/test' ):
    data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225])
])

# TODO: Load the datasets with ImageFolder
image_datasets = {
    'train' : datasets.ImageFolder(train_dir, transform= data_transforms),
    'valid' : datasets.ImageFolder(valid_dir, transform= data_transforms),
    'test' : datasets.ImageFolder(test_dir, transform= data_transforms),
}

# TODO: Using the image datasets and the trainforms, define the dataloaders
batch_size = 8
num_workers = 0
dataloaders = {
    'train' :  torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=True),
    'valid' : torch.utils.data.DataLoader(image_datasets['valid'], batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=False),
    'test' : torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=False)
}
    
    

    