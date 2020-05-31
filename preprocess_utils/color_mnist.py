import os
import math

import torch
import torchvision
import numpy as np
import pandas as pd

from PIL import Image

is_train = True

subdir_str = 'train' if is_train else 'test'
path = os.path.join('data/MNIST/colored', subdir_str)

ds = torchvision.datasets.MNIST('data', train=is_train)
ds_size = len(ds)
red_images = np.random.permutation(ds_size)[:ds_size // 2]
df = pd.DataFrame(np.zeros((ds_size, 2)), columns=['color', 'label'], dtype=int)
df.loc[red_images, 'color'] = 1
df.label = ds.targets.numpy()

os.makedirs(path, exist_ok=True)


def mnist_image_color(im, color_id):
    arr = np.array(im)
    arr = arr.reshape((28, 28, 1))
    arr = np.concatenate([arr, arr, arr], axis=2)
    if color_id != -1:
        for j in range(3):
            if j == color_id:
                continue
            arr[:, :, j] = 0
    return Image.fromarray(arr.astype('uint8'), 'RGB')


for i in range(ds_size):
    im, label = ds[i]
    colored_im = mnist_image_color(im, 0 if i in red_images else -1)
    colored_im.save(os.path.join(path, f'{i}.png'), 'PNG')
df.to_csv(os.path.join(path, 'colors_idx.csv'), index=False)
