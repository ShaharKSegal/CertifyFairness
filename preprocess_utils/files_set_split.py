import os
import shutil

import numpy as np

from collections import namedtuple

root_dir = os.getcwd()
file_ext = ".jpg"
DatasetMetadata = namedtuple("DatasetMetadata", ["dir_name", "size"])

ds_files = [f for f in os.listdir(root_dir) if ".jpg" == os.path.splitext(f)[1]]
ds_size = len(ds_files)

eval_ds = DatasetMetadata("eval_set", int(ds_size * 0.2))
test_ds = DatasetMetadata("test_set", int(ds_size * 0.2))
train_ds = DatasetMetadata("train_set", ds_size - (test_ds.size + eval_ds.size))
ds_lst = [train_ds, eval_ds, test_ds]

ds_permute_indices = np.random.permutation(len(ds_files))
i = 0
for ds_metadata in ds_lst:
    ds_path = os.path.join(root_dir, ds_metadata.dir_name)
    if not os.path.exists(ds_path):
        os.mkdir(ds_path)
    for idx in ds_permute_indices[i:i + ds_metadata.size]:
        file_name = ds_files[idx]
        shutil.move(os.path.join(root_dir, file_name), os.path.join(ds_metadata.dir_name, file_name))
    i += ds_metadata.size
