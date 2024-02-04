import torch
from torchvision import transforms
from torchvision.transforms import functional as vision_F
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
device


def apply_grayscale_with_proba(img: torch.Tensor, p: float):
    random_grayscale = transforms.RandomGrayscale(p=p)
    if len(img.shape) == 3:
        img = random_grayscale(img)
    elif len(img.shape) > 3:
        img = torch.stack([random_grayscale(tensor) for tensor in img])
    
    return img


TRANSFORMS_DICT = [
    ('brightness', vision_F.adjust_brightness, (0.6, 1.4)),
    ('contrast', vision_F.adjust_contrast, (0.6, 1.4)),
    ('saturation', vision_F.adjust_saturation, (0.6, 1.4)),
    ('hue', vision_F.adjust_hue, (-0.1, 0.1)),
    ('gray', apply_grayscale_with_proba, (0, 0.5))
]

def split_interval(lower: float, upper: float, N: int):

    interval_size = (upper - lower) / (N - 1)
    split_points = [round(lower + i * interval_size, 3) for i in range(N)]

    return split_points
    

def get_transforms_list(actions_transform, actions_magnitude, num_magnitudes):
    
    all_transform_lists = []
    for branch in range(actions_transform.size(1)):
        branch_transform_lists = []
        
        for i in range(actions_transform.size(0)):
            transform_list = []

            for s in range(actions_transform.size(2)):
                transform_id = actions_transform[i, branch, s].item()
                magnitude_id = actions_magnitude[i, branch, s].item()
                func_name, func, (lower, upper) = TRANSFORMS_DICT[transform_id]
                magnitudes_list = split_interval(lower, upper, num_magnitudes)
                magnitude = magnitudes_list[magnitude_id]
                transform_list.append((func_name, func, magnitude))
            branch_transform_lists.append(transform_list)
    
        all_transform_lists.append(branch_transform_lists)
        
    return all_transform_lists[0], all_transform_lists[1]


def apply_transformations(img1, transform_list):

    num_samples = img1.size(0)
    stored_imgs = torch.zeros((num_samples, 3, 32, 32))

    for i in range(img1.size(0)):
        img = img1[i]
        for (transform_name, transform_func, magnitude) in transform_list[i]:
            img = transform_func(img, magnitude)
        stored_imgs[i] = img

    return stored_imgs

