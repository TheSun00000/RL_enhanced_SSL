import torch
from torchvision import transforms
from torchvision.transforms import functional as vision_F
import numpy as np
from itertools import permutations

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
device



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

def adjust_brightness(img, *args):
    magnitude = args[0]
    return vision_F.adjust_brightness(img, magnitude)

def adjust_contrast(img, *args):
    magnitude = args[0]
    return vision_F.adjust_contrast(img, magnitude)

def adjust_saturation(img, *args):
    magnitude = args[0]
    return vision_F.adjust_saturation(img, magnitude)

def adjust_hue(img, *args):
    magnitude = args[0]
    return vision_F.adjust_hue(img, magnitude)

def crop_image(img, *args):
    position, scale = args
    return custom_crop(img, position, scale)

def blur_image(img, *args):
    sigma, proba = args
    if torch.rand(1).item() < proba:
        return vision_F.gaussian_blur(img, kernel_size=(3, 3), sigma=sigma)
    return img

def grayscale_image(img, *args):
    proba = args[0]
    if torch.rand(1).item() < proba:
        return vision_F.rgb_to_grayscale(img, num_output_channels=3)
    return img

def flip_image(img, *args):
    return transforms.RandomHorizontalFlip(p=0.5)(img)


TRANSFORMS_DICT = {
    'brightness': {
        'function':adjust_brightness,
        'magnitude':((0.6, 1.4))
    },
    'contrast': {
        'function':adjust_contrast,
        'magnitude':((0.6, 1.4))
    },
    'saturation': {
        'function':adjust_saturation,
        'magnitude':((0.6, 1.4))
    },
    'hue': {
        'function':adjust_hue,
        'magnitude':((-0.1, 0.1))
    },
    'crop': {
        'function':crop_image,
        'position':None,
        'area':(0.2, 1)
    },
    'gray': {
        'function':grayscale_image,
        'proba':(0, 1)
    },
    'blur': {
        'function':blur_image,
        'proba':(0, 1),
        'sigma':(0.1, 2)
    },
    'flip': {
        'function':flip_image
    }
}

permuatations = list(permutations(range(4)))


def split_interval(lower: float, upper: float, N: int):

    interval_size = (upper - lower) / (N - 1)
    split_points = [round(lower + i * interval_size, 3) for i in range(N)]

    return split_points   



def get_transforms_list(actions, num_magnitudes):
    
    all_transform_lists = []
    for branch in range(actions.size(1)):
        branch_transform_lists = []
        
        for i in range(actions.size(0)):
            transform_list = []

            crop_position_index = actions[i, branch, 0]
            crop_area_index = actions[i, branch, 1]
            color_magnitude_index = actions[i, branch, 2:6]
            color_permutation_index = actions[i, branch, 6]
            gray_proba_index = actions[i, branch, 7]
            blur_sigma_index = actions[i, branch, 8]
            blur_proba_index = actions[i, branch, 9]
            
            
            areas_list = split_interval(TRANSFORMS_DICT['crop']['area'][0], TRANSFORMS_DICT['crop']['area'][1], num_magnitudes)
            area = areas_list[crop_area_index]
            position = crop_position_index.item()
            transform_list.append(('crop', TRANSFORMS_DICT['crop']['function'], (position, area)))
            
            transform_list.append(('flip', TRANSFORMS_DICT['flip']['function'], ()))
            
            color_transformations = []
            for color_distortion, magnitude_index in zip(['brightness', 'contrast', 'saturation', 'hue'], color_magnitude_index):
                magnitudes_list = split_interval(TRANSFORMS_DICT[color_distortion]['magnitude'][0], TRANSFORMS_DICT[color_distortion]['magnitude'][1], num_magnitudes)
                magnitude = magnitudes_list[magnitude_index]
                color_transformations.append((color_distortion, TRANSFORMS_DICT[color_distortion]['function'], (magnitude,)))
            color_transformations = [ color_transformations[i] for i in permuatations[color_permutation_index]]
            transform_list += color_transformations
            
            probas_list = split_interval(TRANSFORMS_DICT['gray']['proba'][0], TRANSFORMS_DICT['gray']['proba'][1], num_magnitudes)
            proba = probas_list[gray_proba_index]
            transform_list.append(('gray', TRANSFORMS_DICT['gray']['function'], (proba,)))
            
            sigmas_list = split_interval(TRANSFORMS_DICT['blur']['sigma'][0], TRANSFORMS_DICT['blur']['sigma'][1], num_magnitudes)
            probas_list = split_interval(TRANSFORMS_DICT['blur']['proba'][0], TRANSFORMS_DICT['blur']['proba'][1], num_magnitudes)
            sigma = sigmas_list[blur_sigma_index]
            proba = probas_list[blur_proba_index]
            transform_list.append(('blur', TRANSFORMS_DICT['blur']['function'], (sigma, proba)))
            
            
            branch_transform_lists.append(transform_list)
    
        all_transform_lists.append(branch_transform_lists)
        
    return all_transform_lists[0], all_transform_lists[1]



def apply_transformations(img1, transform_list):

    num_samples = img1.size(0)
    stored_imgs = torch.zeros((num_samples, 3, 32, 32))

    for i in range(img1.size(0)):
        img = img1[i]
        for (transform_name, transform_func, args) in transform_list[i]:
            img = transform_func(img, *args)
        stored_imgs[i] = img

    return stored_imgs




