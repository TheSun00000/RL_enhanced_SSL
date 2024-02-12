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
    ('gray', apply_grayscale_with_proba, (0, 0.2))
]

def split_interval(lower: float, upper: float, N: int):

    interval_size = (upper - lower) / (N - 1)
    split_points = [round(lower + i * interval_size, 3) for i in range(N)]

    return split_points
    

def clip(n, min, max):
    if n > max:
        return max
    elif n < min:
        return min
    return n

def custom_crop(img, position, scale):
    """_summary_

    Args:
        position (_type_): in 
            [0, 1, 2,
             3, 4, 5,
             6, 7, 8]
        target_area (float): in [0.08, 1.]
    """

    position = clip(position, 0, 8)
    
    ratio = (3.0 / 4.0, 4.0 / 3.0)
    
    *_, height, width = img.shape
    area = height * width

    target_area = area * scale
    log_ratio = torch.log(torch.tensor(ratio))
    aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

    
    w = int(round((target_area * aspect_ratio)**0.5))
    h = int(round((target_area / aspect_ratio)**0.5))
    h = clip(h, 0, height)
    w = clip(w, 0, width)
    
    i, j = height * (position//3)/3, width * (position%3)/3
    i_noise = torch.empty(1).uniform_(0, height/3).item()
    j_noise = torch.empty(1).uniform_(0, width/3).item()    
    i, j = int(round(i + i_noise)), int(round(j + j_noise))
    
    if i + h > height:
        i = height - h
    if j + w > width:
        j = width - w 
    
    return vision_F.resized_crop(img, i, j, h, w, size=(height, width), antialias="warn")
    


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
        # print('----', img.min(), img.max())
        for (transform_name, transform_func, magnitude) in transform_list[i]:
            img = transform_func(img, magnitude)
            # print('----', transform_name, img.min(), img.max())
        stored_imgs[i] = img

    return stored_imgs




