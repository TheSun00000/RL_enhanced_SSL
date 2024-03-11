import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from utils.networks import DecoderNN_1input
from utils.transforms import (
    apply_transformations,
    get_transforms_list,
    RandomAugmentation,
    Augmentation
)

import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device


import matplotlib.pyplot as plt
import numpy as np

def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).unsqueeze(1).unsqueeze(2)
    std = torch.tensor(std).unsqueeze(1).unsqueeze(2)
    denormalized_tensor = tensor * std + mean
    denormalized_tensor.clamp_(0, 1)
    return denormalized_tensor


def plot_images_stacked(tensor1, tensor2):
    
    _, _, h, w = tensor1.shape
    
    tensor1 = tensor1.cpu()
    tensor2 = tensor2.cpu()
    # Check if the input tensors have the correct shape
    expected_shape = (3, h, h)
    if tensor1.shape[1:] != expected_shape or tensor2.shape[1:] != expected_shape:
        raise ValueError("Input tensors must have shape (N, 3, H, W)")

    # Denormalize tensors
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    tensor1 = denormalize(tensor1, mean, std)
    tensor2 = denormalize(tensor2, mean, std)

    # Set up the figure for subplots
    fig, axes = plt.subplots(2, tensor1.shape[0], figsize=(tensor1.shape[0]*4, 8))

    # Plot images from the first tensor
    for image_index in range(tensor1.shape[0]):
        image_data = tensor1[image_index].permute(1, 2, 0)  # Transpose to (32, 32, 3) for RGB
        axes[0, image_index].imshow(image_data)
        axes[0, image_index].axis('off')

    # Plot images from the second tensor
    for image_index in range(tensor2.shape[0]):
        image_data = tensor2[image_index].permute(1, 2, 0)  # Transpose to (32, 32, 3) for RGB
        axes[1, image_index].imshow(image_data)
        axes[1, image_index].axis('off')

    plt.show()


def MyRawDatset_collate_fn(imgs_y):
    imgs, targets = [], []
    for img, y in imgs_y:
        imgs.append(img)
        targets.append(y)
        
    targets = torch.tensor(targets, dtype=torch.long)
    return imgs, targets


class MyDatset(Dataset):
    def __init__(self, train_dataset, policies=[], random_p=1, ppo_dist=[], transform=True, normalize=None, random_resized_crop=None):
                
        self.train_dataset = train_dataset
        self.policies = policies
        self.random_p = random_p
        self.ppo_dist = ppo_dist
        self.transform = transform
        
        self.random_policy = RandomAugmentation(N=2, pr=0.8)
        self.ppo_policy = None
        if random_p != 1:
            self.ppo_policy = Augmentation(policies, dist=ppo_dist)
        
        self.last_transform = transforms.Compose([
            random_resized_crop,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize
        ])


    def __len__(self,):
        return len(self.train_dataset)
    
    def __getitem__(self, i):
        img, y = self.train_dataset[i][0], self.train_dataset[i][1]
        
        if not self.transform:
            return img, y
        
        img1, img2 = img.copy(), img.copy()
        
        img1_is_random = random.random() < self.random_p
        img2_is_random = random.random() < self.random_p
        
        if img1_is_random and img2_is_random:
            img1 = self.random_policy(img1)
            img2 = self.random_policy(img2)
        
        elif not img1_is_random and img2_is_random:
            img1 = self.ppo_policy(img1, branch=1)
            img2 = self.random_policy(img2)
        
        elif img1_is_random and not img2_is_random:
            img1 = self.random_policy(img1)
            img2 = self.ppo_policy(img2, branch=2)
        
        elif not img1_is_random and not img2_is_random:
            img1, img2 = self.ppo_policy(img) 
        
        img = self.last_transform(img)
        img1 = self.last_transform(img1)
        img2 = self.last_transform(img2)
        y = torch.tensor(y, dtype=torch.long).unsqueeze(0)
        
        return img, img1, img2, y



def get_dataloader(dataset_name, batch_size, policies=[], random_p=1, ppo_dist=[], transform=True):        
    
    if dataset_name in ['cifar10', 'svhn'] :
        normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        random_resized_crop = transforms.RandomResizedCrop(32, scale=(0.2, 1.))
        
        if dataset_name == 'cifar10':
            dataset = torchvision.datasets.CIFAR10('./dataset/')
        elif dataset_name == 'svhn':
            dataset = torchvision.datasets.SVHN('./dataset/SVHN/', split='train')
        
        dataset = MyDatset(
            train_dataset=dataset,
            policies=policies,
            random_p=random_p,
            ppo_dist=ppo_dist,
            transform=transform,
            normalize=normalize,
            random_resized_crop=random_resized_crop)
        

    elif dataset_name == 'TinyImagenet':
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        random_resized_crop = transforms.RandomResizedCrop(64, scale=(0.2, 1.))

        dataset = MyDatset(
            train_dataset=torchvision.datasets.ImageFolder('dataset/tiny-imagenet-200/train'),
            policies=policies,
            random_p=random_p,
            ppo_dist=ppo_dist,
            transform=transform,
            normalize=normalize,
            random_resized_crop=random_resized_crop
        )
    
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        collate_fn=None if transform else MyRawDatset_collate_fn,
        pin_memory=True
    )

    
    return data_loader




def get_knn_evaluation_loader(dataset_name, batch_size=512):
    
    if dataset_name in ['cifar10', 'svhn']:
        single_transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])

        if dataset_name == 'cifar10':
            train_dataset = torchvision.datasets.CIFAR10('./dataset', train=True, transform=single_transform)
            test_dataset = torchvision.datasets.CIFAR10('./dataset', train=False, transform=single_transform)
        elif dataset_name == 'svhn':
            train_dataset = torchvision.datasets.SVHN('./dataset/SVHN/', split='train', transform=single_transform)
            test_dataset = torchvision.datasets.SVHN('./dataset/SVHN/', split='test', transform=single_transform)
    
    
    elif dataset_name == 'TinyImagenet':
        single_transform = transforms.Compose([
            # transforms.Resize(32),
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_dataset = torchvision.datasets.ImageFolder('./dataset/tiny-imagenet-200/train', transform=single_transform)
        test_dataset = torchvision.datasets.ImageFolder('./dataset/tiny-imagenet-200/val/images', transform=single_transform)
        
        
        
    
    memory_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=1
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=1
    )
        
    return memory_loader, test_loader



def get_linear_evaluation_loader(dataset_name, batch_size):
    
    if dataset_name in ['cifar10', 'svhn']:

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(36, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])

        if dataset_name == 'cifar10':
            train_dataset = torchvision.datasets.CIFAR10('./dataset', train=True, transform=train_transform)
            test_dataset = torchvision.datasets.CIFAR10('./dataset', train=False, transform=test_transform)
        elif dataset_name == 'svhn':
            train_dataset = torchvision.datasets.SVHN('./dataset/SVHN/', split='train', transform=train_transform)
            test_dataset = torchvision.datasets.SVHN('./dataset/SVHN/', split='test', transform=test_transform)
    

    
    elif dataset_name == 'TinyImagenet':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(64, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(72, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])

        train_dataset = torchvision.datasets.ImageFolder('./dataset/tiny-imagenet-200/train', transform=train_transform)
        test_dataset = torchvision.datasets.ImageFolder('./dataset/tiny-imagenet-200/val/images', transform=test_transform)
        
        
        
    
    memory_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=1
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=1
    )
        
    return memory_loader, test_loader