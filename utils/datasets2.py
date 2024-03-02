import torch
import torchvision
import torchvision.transforms as T

normalize = T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
single_transform = T.Compose([T.ToTensor(), normalize])

class ContrastiveLearningTransform:
    def __init__(self):
        transforms = [
            T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2)
        ]
        transforms_rotation = [
            T.RandomResizedCrop(size=16, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2)
        ]

        self.transform = T.Compose(transforms)
        self.transform_rotation = T.Compose(transforms_rotation)

    def __call__(self, x):
        # output = [
        #     single_transform(self.transform(x)),
        #     single_transform(self.transform(x)),
        #     single_transform(self.transform_rotation(x))
        # ]
        # return output
        return single_transform(self.transform(x))
    
    
train_loader = torch.utils.data.DataLoader(
    dataset=torchvision.datasets.CIFAR10(
        'dataset', train=True, transform=ContrastiveLearningTransform(), download=True
    ),
    shuffle=True,
    batch_size=512,
    pin_memory=True,
    # num_workers=1,
    drop_last=True
)
memory_loader = torch.utils.data.DataLoader(
    dataset=torchvision.datasets.CIFAR10(
        'dataset', train=True, transform=single_transform, download=True
    ),
    shuffle=False,
    batch_size=512,
    pin_memory=True,
    num_workers=1,
)
test_loader = torch.utils.data.DataLoader(
    dataset=torchvision.datasets.CIFAR10(
        'dataset', train=False, transform=single_transform, download=True,
    ),
    shuffle=False,
    batch_size=512,
    pin_memory=True,
    num_workers=1
)


def get_essl_train_loader():
    return train_loader

def get_essl_memory_loader():
    return memory_loader

def get_essl_test_loader():
    return test_loader