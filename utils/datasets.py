import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from utils.networks import DecoderNoInput, DecoderRNN
from utils.transforms import (
    apply_transformations,
    get_transforms_list
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
device






cifar10_dataset = torchvision.datasets.CIFAR10('dataset', download=True)


class MyDatset(Dataset):
    def __init__(self, train_dataset):
        self.train_dataset = train_dataset
        self.to_tensor = transforms.ToTensor()

    def __len__(self,):
        return len(self.train_dataset)
    
    def __getitem__(self, i):
        img = self.train_dataset[i][0]
        x1 = self.to_tensor(img)
        x2 = self.to_tensor(img)
        return x1, x2
    
    

class DataLoaderWrapper:
    def __init__(self, dataloder, steps, encoder, decoder, random_p, spatial_only):
        self.dataloder = dataloder
        self.steps = steps
        self.encoder = encoder
        self.decoder = decoder
        self.random_p = random_p
        self.spatial_only = spatial_only
        
        self.random_transformation = self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
            ])
        
        self.random_spatial_transformation = self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
    
    
    def decoder_transform(self, x):
        
        batch_size = x[0].shape[0]
        num_random_samples = int(batch_size*self.random_p)
        num_decoder_samples = batch_size - num_random_samples
        
        x1, x2 = x        
        
        if self.spatial_only:
            x1 = torch.stack([self.random_spatial_transformation(tensor) for tensor in x1])
            x2 = torch.stack([self.random_spatial_transformation(tensor) for tensor in x2])
            return x1, x2
        
        
        random_x1, decoder_x1 = x1[:num_random_samples], x1[num_random_samples:]
        random_x2, decoder_x2 = x2[:num_random_samples], x2[num_random_samples:]

        if num_random_samples != 0:
            random_x1 = torch.stack([self.random_transformation(tensor) for tensor in random_x1])
            random_x2 = torch.stack([self.random_transformation(tensor) for tensor in random_x2])
        
        if num_decoder_samples != 0:
            decoder_x1 = torch.stack([self.random_spatial_transformation(tensor) for tensor in decoder_x1])
            decoder_x2 = torch.stack([self.random_spatial_transformation(tensor) for tensor in decoder_x2])


        if (num_decoder_samples != 0):
            if isinstance(self.decoder, DecoderRNN):
                decoder_x1 = decoder_x1.to(device)
                decoder_x2 = decoder_x2.to(device)
                with torch.no_grad():
                    _, z1 = self.encoder(decoder_x1)
                    _, z2 = self.encoder(decoder_x2)
                    (_, (transform_actions_index, magnitude_actions_index), _) = self.decoder(z1, z2)
                    
            elif isinstance(self.decoder, DecoderNoInput):
                with torch.no_grad():
                    (_, (transform_actions_index, magnitude_actions_index), _) = self.decoder(num_decoder_samples)
            
            
            transforms_list_1, transforms_list_2 = get_transforms_list(transform_actions_index, magnitude_actions_index)
            decoder_x1 = apply_transformations(decoder_x1.cpu(), transforms_list_1)
            decoder_x2 = apply_transformations(decoder_x2.cpu(), transforms_list_2)
                    
        
        new_x1 = torch.cat((random_x1, decoder_x1))
        new_x2 = torch.cat((random_x2, decoder_x2))
              
        return (new_x1, new_x2)
        
    
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
                    x = next(iterator)
                except StopIteration:
                    iterator = iter(self.dataloder)
                    x = next(iterator)
                
                x = self.decoder_transform(x)
                yield x
                
        else: # self.steps in ['all', -1]
            for x in iterator:
                x = self.decoder_transform(x)
                yield x
          


def get_cifar10_dataloader(num_steps, batch_size, encoder=None, decoder=None, random_p=0, spatial_only=False):
        
    dataset = MyDatset(cifar10_dataset)
    data_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    wrapped_data_loader = DataLoaderWrapper(data_loader, num_steps, encoder=encoder, decoder=decoder, random_p=random_p, spatial_only=spatial_only)

    return wrapped_data_loader