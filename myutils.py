import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch.utils.data
from torch.utils.data import Dataset

import os.path

class ApplyTransform(Dataset):
    def __init__(self, dataset: Dataset, transform: torch.nn.Module=None, target_transform: torch.nn.Module=None):
        self.dataset: Dataset = dataset
        self.transform: torch.nn.Module = transform
        self.target_transform: torch.nn.Module = target_transform
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample, target = self.dataset[idx]
        if self.transform is not None:
            sample: torch.Tensor = self.transform(sample)
        if self.target_transform is not None:
            target: torch.Tensor = self.target_transform(target)
        return sample, target
    def __len__(self) -> int:
        return len(self.dataset)

def load_data(data_root: str='./dataset', train_val_ratio: float=0.9) -> tuple[Dataset, Dataset, Dataset, Dataset]:
    dset: Dataset=CIFAR10; num_classes: int=10; num_examples: int=50000
    # transforms
    mean: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    std:  tuple[float, float, float] = (0.2023, 0.1994, 0.2010)
    transform_train: torch.nn.Module = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(x.unsqueeze(0), (4, 4, 4, 4), mode='reflect').squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize( mean, std )
    ])
    transform_test: torch.nn.Module = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize( mean, std )
    ])
    # datasets
    trainvalset: Dataset = dset(
        root=data_root, train=True, download=True, transform=None)
    testset: Dataset = dset(
        root=data_root, train=False, download=True, transform=transform_test)
    num_examples_train: int = int(len(trainvalset)*train_val_ratio)
    num_examples_val: int = int(len(trainvalset)) - num_examples_train
    trainset, valset = torch.utils.data.random_split(
        trainvalset, [num_examples_train, num_examples_val] )
    # transformations, including several data augmentations for training
    trainvalset: Dataset = ApplyTransform(trainvalset, transform=transform_train)
    trainset:    Dataset = ApplyTransform(trainset,    transform=transform_train)
    valset:      Dataset = ApplyTransform(valset,      transform=transform_test)
    #
    return trainvalset, trainset, valset, testset

def save_model(save_dir: str, filename: str, net: torch.nn.Module, optimizer: torch.optim.Optimizer,
               losses: list[float], accs: list[float], num_epochs: int) -> None:
    path: str = os.path.join(save_dir, filename)
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': losses[-1],
        'val_accs': accs[-1],
    }, path)
