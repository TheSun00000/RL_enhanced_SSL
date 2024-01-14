import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from utils.transforms import (
    apply_transformations,
    get_transforms_list
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
device






cifar10_dataset = torchvision.datasets.CIFAR10('dataset', download=True)


class MyDatset(Dataset):
    def __init__(self, train_dataset, transform):
        self.train_dataset = train_dataset
        
        if transform == True:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                # transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                # normalize,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                # normalize,
            ])

    def __len__(self,):
        return len(self.train_dataset)
    
    def __getitem__(self, i):
        img = self.train_dataset[i][0]
        x1 = self.transform(img)
        x2 = self.transform(img)
        return x1, x2
    
    

class DataLoaderWrapper:
    def __init__(self, dataloder, steps, encoder, decoder):
        self.dataloder = dataloder
        self.steps = steps
        self.encoder = encoder
        self.decoder = decoder
    
    
    def decoder_transform(self, x):
        
        x1, x2 = x
        x1 = x1.to(device)
        x2 = x2.to(device)

        with torch.no_grad():
            _, z1 = self.encoder(x1)
            _, z2 = self.encoder(x2)

            ((transform_actions_index, _), (magnitude_actions_index, _), _) = self.decoder(z1, z2)

            transforms_list_1, transforms_list_2 = get_transforms_list(transform_actions_index, magnitude_actions_index)
            
            x1 = x1.cpu()
            x2 = x2.cpu()
            
            new_x1 = apply_transformations(x1, transforms_list_1)
            new_x2 = apply_transformations(x2, transforms_list_2)
                
        return (new_x1, new_x2)
    
    
    def __len__(self):
        if self.steps in ['all', -1]:
            return len(self.dataloder)
        else:
            return self.steps
    
    def __iter__(self):
        
        transform = (self.encoder is not None) and (self.decoder is not None)
        
        iterator = iter(self.dataloder)
        if not self.steps in ['all', -1]:
            for i in range(self.steps):
                try:
                    x = next(iterator)
                except StopIteration:
                    iterator = iter(self.dataloder)
                    x = next(iterator)
                
                if transform:
                    x = self.decoder_transform(x)
                yield x
                
        else: # self.steps in ['all', -1]
            for x in iterator:
                if transform:
                    x = self.decoder_transform(x)
                yield x
        


def get_cifar10_dataloader(num_steps, batch_size, transform=False, encoder=None, decoder=None):
    ppo_transform = (encoder is not None) and (decoder is not None)   
    
    assert ~transform and ~ppo_transform, 'cant use both random_transform and ppo_transform'
    
    
    dataset = MyDatset(cifar10_dataset, transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    wrapped_data_loader = DataLoaderWrapper(data_loader, num_steps, encoder=encoder, decoder=decoder)

    return wrapped_data_loader