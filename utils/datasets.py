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
    tensor1 = tensor1.cpu()
    tensor2 = tensor2.cpu()
    # Check if the input tensors have the correct shape
    expected_shape = (3, 32, 32)
    if tensor1.shape[1:] != expected_shape or tensor2.shape[1:] != expected_shape:
        raise ValueError("Input tensors must have shape (N, 3, 32, 32)")

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


def rotate_images(images):
    nimages = images.shape[0]
    n_rot_images = 4 * nimages

    # rotate images all 4 ways at once
    rotated_images = torch.zeros([nimages, 4, images.shape[1], images.shape[2], images.shape[3]])
    rot_classes = torch.zeros([nimages, 4]).long()

    rotated_images[:, 0] = images
    # rotate 90
    rotated_images[:, 1] = images.flip(3).transpose(2, 3)
    rot_classes[:, 1] = 1
    # rotate 180
    rotated_images[:, 2] = images.flip(3).flip(2)
    rot_classes[:, 2] = 2
    # rotate 270
    rotated_images[:, 3] = images.transpose(2, 3).flip(3)
    rot_classes[:, 3] = 3

    rotated_images = rotated_images.reshape(-1, images.shape[1], images.shape[2], images.shape[3])
    rot_classes = rot_classes.reshape(-1)
    
    return rotated_images, rot_classes


def select(list, ids):
    return [list[i] for i in ids]



cifar10_dataset = torchvision.datasets.CIFAR10('dataset', download=True)

    
    

class MyRawDatset(Dataset):
    def __init__(self, train_dataset):
        self.train_dataset = train_dataset

    def __len__(self,):
        return len(self.train_dataset)
    
    def __getitem__(self, i):
        img, y = self.train_dataset[i][0], self.train_dataset[i][1]
        return img, y


def MyRawDatset_collate_fn(imgs_y):
    imgs, targets = [], []
    for img, y in imgs_y:
        imgs.append(img)
        targets.append(y)
        
    targets = torch.tensor(targets, dtype=torch.long)
    return imgs, targets


class DataLoaderWrapper:
    def __init__(self, dataloder, steps):
        self.steps = steps
        self.dataloder = dataloder
    
    def __len__(self):
        if self.steps in ['all', -1]:
            return len(self.dataloder)
        else:
            return self.steps
    
    def __iter__(self):
                
        iterator = iter(self.dataloder)
        if not self.steps in ['all', -1]:
            for i in range(self.steps):
                try:
                    x, y = next(iterator)
                except StopIteration:
                    iterator = iter(self.dataloder)
                    x, y = next(iterator)
                yield x, y
                
        else: # self.steps in ['all', -1]
            for x, y in iterator:
                yield x, y
          



class MyDatset(Dataset):
    def __init__(self, train_dataset, policies, random_p, ppo_dist):
        self.train_dataset = train_dataset
        self.policies = policies
        self.random_p = random_p
        
        self.random_policy = RandomAugmentation(N=3, pr=0.8)
        self.ppo_policy = None
        if random_p != 1:
            self.ppo_policy = Augmentation(policies, dist=ppo_dist)
        
        self.last_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])


    def __len__(self,):
        return len(self.train_dataset)
    
    def __getitem__(self, i):
        img, y = self.train_dataset[i][0], self.train_dataset[i][1]
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





def get_cifar10_dataloader(batch_size, random_p, all_policies, ppo_dist):
    
    dataset = MyDatset(cifar10_dataset, all_policies, random_p, ppo_dist)
    data_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    return data_loader

def get_cifar10_raw_dataloader(num_steps, batch_size):
    
    dataset = MyRawDatset(cifar10_dataset)
    data_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True, collate_fn=MyRawDatset_collate_fn)
    data_loader = DataLoaderWrapper(data_loader, num_steps)

    return data_loader


def select_from_rotated_views(rotated_x1, rotated_x2, rotated_labels1, rotated_labels2):
    
    
    batch_size = rotated_x1.shape[0] // 4
    
    # Select images from both img1 and img2
    rotated_x1, rotated_x2 = rotated_x1.reshape(-1, 4, 3, 32, 32), rotated_x2.reshape(-1, 4, 3, 32, 32)
    rotated_labels1, rotated_labels2 = rotated_labels1.reshape(-1, 4), rotated_labels2.reshape(-1, 4)
        
    rotated_x = torch.concat((rotated_x1, rotated_x2), dim=1)
    rotated_labels = torch.concat((rotated_labels1, rotated_labels2), dim=1)
    
    selected_idx = torch.zeros((batch_size, 4), dtype=torch.long)
    for i in range(batch_size):
        selected_idx[i] = torch.randperm(8)[:4]
    rotated_x = rotated_x[torch.arange(batch_size).unsqueeze(1), selected_idx]
    rotated_labels = rotated_labels[torch.arange(batch_size).unsqueeze(1), selected_idx]
    
    rotated_x, rotated_labels = rotated_x.reshape(-1, 3, 32, 32), rotated_labels.reshape(-1)            
    
    
    return rotated_x, rotated_labels