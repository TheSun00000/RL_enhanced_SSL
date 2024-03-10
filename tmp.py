import pickle
import os
from PIL import Image

from tqdm import tqdm
import numpy as np


# with open('dataset/imagenet100_train.pkl', 'rb') as file:
#     to_save = pickle.load(file)

# key = list(to_save.keys())[0]
# img_path, img, y = to_save[key]

# print(img)
# print(key)
# print(img_path)
# print(y)


# exit()

path = 'dataset/imagenet100/train'
classes = os.listdir(path)

class_dict = {}
img_path_class = []
for c_i, c in enumerate(classes):
    class_dict[c] = c_i
    for img_path in os.listdir(f'dataset/imagenet100/train/{c}/'):
        img = img_path.split('.')[0]
        img_path_class.append((img, f'dataset/imagenet100/train/{c}/{img_path}', c))
        




# with open('dataset/imagenet100_train.pkl', 'rb') as file:
#     data = pickle.load(file)
#     if data:
#         to_save = data



    
# for i in tqdm(range(len(img_path_class))):
#     img_name, img_path, class_ = img_path_class[i]

#     img = Image.open(img_path)
#     if img.mode != 'RGB':
#         img = img.convert("RGB")
#     # img = img.resize((224, 224))
#     img = np.array(img)
    
#     if not class_ in os.listdir('dataset/imagenet100/numpy/'):
#         os.mkdir(f'dataset/imagenet100/numpy/{class_}/')
    
#     np.save(f'dataset/imagenet100/numpy/{class_}/{img_name}.npy', img)

#     # if i == 1000:
#     #     break
            
            
         
# with open('dataset/imagenet100_train.pkl', 'wb') as file:
#             pickle.dump(to_save, file)