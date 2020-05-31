import os

import numpy as np
import pandas as pd

from PIL import Image

debug = False
debug_pic_num = 100

dataset_path = os.path.join("..", "data", "MNIST", "colored")
images_subdir = 'train'
metadata_file = os.path.join("train", "colors_idx.csv") #'celeba_metadata.csv'

metadata = pd.read_csv(os.path.join(dataset_path, metadata_file))
metadata_train = metadata

# faces_file_strs = os.listdir(dataset_path)
# pics_num = debug_pic_num if debug else len(faces_file_strs)
# faces_file_strs = faces_file_strs[:pics_num]
pics_num = debug_pic_num if debug else metadata_train.shape[0]
metadata_train = metadata_train.iloc[:pics_num]
mean_channel_dataset = np.zeros(3)
std_channel_dataset = np.zeros(3)

# calc mean
# for i, face_pic_str in enumerate(faces_file_strs):
# image_full_path = os.path.join(dataset_path, face_pic_str)
for idx, row in metadata_train.iterrows():
    image_full_path = os.path.join(dataset_path, images_subdir, f"{idx}.png")
    img = Image.open(image_full_path)
    pic_arr = np.array(img)
    mean_channel_dataset += (pic_arr / 225.).mean(axis=(0, 1))
mean_channel_dataset /= pics_num

# calc std
for idx, row in metadata_train.iterrows():
    image_full_path = os.path.join(dataset_path, images_subdir, f"{idx}.png")
    img = Image.open(image_full_path)
    pic_arr = np.array(img)
    std_channel_dataset += ((pic_arr / 225.) ** 2).mean(axis=(0, 1)) - mean_channel_dataset ** 2
std_channel_dataset /= pics_num
std_channel_dataset = np.sqrt(std_channel_dataset)

print(mean_channel_dataset, std_channel_dataset)
