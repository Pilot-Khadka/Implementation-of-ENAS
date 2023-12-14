import yaml
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

batch_size =32

all_transforms = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()
]
)

training_data = datasets.FashionMNIST(
    root = "./data",
    train = True,
    download = True,
    transform = all_transforms
)

test_data = datasets.FashionMNIST(
    root="./data",
    train = False,
    download = True,
    transform = all_transforms
)

train_loader = torch.utils.data.DataLoader(dataset = training_data,
                                           batch_size = batch_size,
                                           shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_data,
                                          batch_size = batch_size,
                                          shuffle = True)
class Config:
    def __init__(self, file_path = 'parameters.yaml'):
        with open(file_path,"r") as yaml_file:
            self.parameters = yaml.safe_load(yaml_file)

