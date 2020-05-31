import os
import shutil

import numpy as np
import pandas as pd

from collections import namedtuple
from PIL import Image

src_dir = os.path.join("..", "data", "lfw")

# load attribute file
df = pd.read_csv(os.path.join(src_dir, 'lfw_attributes.txt'), delimiter='\t', skiprows=1)
# binarize all attributes
df[df.columns[2:]] = np.clip(np.sign(df[df.columns[2:]]), 0, 1)
# fix extra column problem
data = df.iloc[:, : -1]
data.columns = df.columns[1:]


ds_size = data.shape[0]
ds_permute_indices = np.random.permutation(ds_size)
ds_train_indices = ds_permute_indices[:int(0.8*ds_size)]
ds_test_indices = ds_permute_indices[int(0.8*ds_size):]

ds_train_indices = np.sort(ds_train_indices)
ds_test_indices = np.sort(ds_test_indices)

np.savetxt(os.path.join(src_dir, 'train_indices.txt'), ds_train_indices, fmt='%i')
np.savetxt(os.path.join(src_dir, 'test_indices.txt'), ds_test_indices, fmt='%i')
