from torch.utils.data import DataLoader, Dataset
import torchvision
from utils.datasets import MyDatset
import os
from PIL import Image

from tqdm import tqdm

import numpy as np
import datetime
import torch


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


# in100_dataset_train = torchvision.datasets.ImageFolder(root='dataset/imagenet100/train')
# in100_dataset_val = torchvision.datasets.ImageFolder(root='dataset/imagenet100/val')




class MyImageNet100Dataset(Dataset):
    def __init__(self, path, num_classes, num_img_per_class) -> None:
        super().__init__()
        self.path = path
        self.num_classes = num_classes
        self.classes = os.listdir(path)[:num_classes]
        
        # self.img_class_dict = {}
        self.class_dict = {}
        self.img_path_class = []
        self.class_img_dict = {}
        for c_i, c in enumerate(self.classes):
            self.class_dict[c] = c_i
            self.class_img_dict[c] = []
            for img_path in os.listdir(f'dataset/imagenet100/train/{c}/')[:num_img_per_class]:
                img = img_path.split('.')[0]
                # self.img_class_dict[img] = (f'dataset/imagenet100/train/{c}/{img_path}', c)
                self.img_path_class.append((img, f'dataset/imagenet100/train/{c}/{img_path}', c))
        
        
        # self.all_images = []
        
        # s1 = []
        # s2 = []
        
        # for index in tqdm(range(len(self.img_path_class))):
        #     img_name, _, y = self.img_path_class[index]
        #     img_path = f'dataset/imagenet100/numpy/{y}/{img_name}.npy'
        #     # print(img_path)
            
        #     if len(self.class_img_dict[y]) < num_img_per_class:
        #         img = np.load(img_path)
        #         img = Image.fromarray(img)
                
        #         self.class_img_dict[y].append(0)
                
        #         # self.all_images.append((img, self.class_dict[y]))
        #         self.all_images.append(img.size)
        #         a, b = img.size
        #         s1.append(a)
        #         s2.append(b)
                            
        # print(sum(s1)/len(s1))
        # print(sum(s2)/len(s2))
        
        self.randocrop = transforms.RandomResizedCrop(size=224, scale=(0.2, 1.))
        self.random_policy = RandomAugmentation(N=2, pr=0.8)
        self.last_transform = transforms.Compose([
            # transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        
        
    def __len__(self):
        return len(self.img_path_class)
    
    
    def __getitem__(self, index):
        img_name, _, y = self.img_path_class[index]
        img_path = f'dataset/imagenet100/numpy/{y}/{img_name}.npy'
        img = np.load(img_path)
        img = Image.fromarray(img)
        
        
        x1 = self.randocrop(img)
        x1 = self.random_policy(x1)
        x1 = self.last_transform(x1)
        
        x2 = self.randocrop(img)
        x2 = self.random_policy(x2)
        x2 = self.last_transform(x2)
        
        y = torch.tensor(self.class_dict[y], dtype=torch.long).unsqueeze(0)
        
        return x1, x2, y
        

# cifar10_dataset = torchvision.datasets.CIFAR10('dataset', download=True)
in100_dataset_train = MyImageNet100Dataset('dataset/imagenet100/train', 100, 1000)
in100_dataset_train = DataLoader(in100_dataset_train, 512, shuffle=True, num_workers=0)

# x, y = cifar10_dataset[0]
# print(x, y)
# x, y = in100_dataset_train[0]
# print(x, y)




# exit()

# dataset = MyDatset(
#     train_dataset=in100_dataset_train,
#     policies=[],
#     random_p=1,
#     ppo_dist=[]
# )

# loader = DataLoader(dataset, batch_size=512, drop_last=True, shuffle=True, num_workers=0)

# print(len(loader))

# num_threads = torch.get_num_threads()
# print(num_threads)

# for i in tqdm(loader, total=len(loader)):
#     pass

for i in tqdm(in100_dataset_train):
    pass

# print(len(dataset))

# img, y = dataset[100]

# print(img, y)