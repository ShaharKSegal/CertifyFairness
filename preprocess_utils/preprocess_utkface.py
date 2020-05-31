import os
import shutil

import numpy as np
import pandas as pd

from collections import namedtuple
from PIL import Image

src_dir = os.path.join(".", "UTKFace")
dest_dir = os.path.join("..", "data", "UTKFace")
race_filter = [0, 1]


def create_sets(source_dir, destination_dir, override=False):
    file_ext = ".jpg"
    DatasetMetadata = namedtuple("DatasetMetadata", ["dir_name", "size"])

    ds_files = [f for f in os.listdir(source_dir) if file_ext == os.path.splitext(f)[1]]
    face_pic_lst = []

    for face_pic_str in ds_files:
        face_pic_prop = face_pic_str.split('_')
        if len(face_pic_prop) != 4:
            continue
        age, gender, race, date = face_pic_prop
        date = date[:date.index('.')]
        if race_filter is not None and int(race) not in race_filter:
            continue
        face_pic_lst.append([face_pic_str, date, age, gender, race])

    face_pic_df = pd.DataFrame(face_pic_lst, columns=['file_name', 'date', 'age', 'gender', 'race'])
    face_pic_df.astype({"file_name": str, "date": str, "age": int, "gender": int, "race": int})

    ds_size = face_pic_df.shape[0]

    eval_ds = DatasetMetadata("eval_set", int(ds_size * 0.5))
    train_ds = DatasetMetadata("train_set", ds_size - (eval_ds.size))
    ds_lst = [train_ds, eval_ds]
    # test_ds = DatasetMetadata("test_set", int(ds_size * 0.2))
    # train_ds = DatasetMetadata("train_set", ds_size - (test_ds.size + eval_ds.size))
    # ds_lst = [train_ds, eval_ds, test_ds]

    ds_permute_indices = np.random.permutation(ds_size)
    i = 0
    for ds_metadata in ds_lst:
        ds_path = os.path.join(destination_dir, ds_metadata.dir_name)
        if os.path.exists(ds_path):
            if override:
                shutil.rmtree(ds_path)
            else:
                print("Directory", ds_path, "already exists. Set override=True to force deletion.")
        os.makedirs(ds_path, exist_ok=True)
        for idx in ds_permute_indices[i:i + ds_metadata.size]:
            file_name = face_pic_df.iloc[idx].file_name
            shutil.copy2(os.path.join(source_dir, file_name), os.path.join(ds_path, file_name))
        i += ds_metadata.size

    dataset_path = os.path.join(destination_dir, "train_set")

    faces_file_strs = os.listdir(dataset_path)
    pics_num = len(faces_file_strs)
    faces_file_strs = faces_file_strs[:pics_num]
    mean_channel_dataset = np.zeros(3)
    std_channel_dataset = np.zeros(3)

    # calc mean
    for i, face_pic_str in enumerate(faces_file_strs):
        image_full_path = os.path.join(dataset_path, face_pic_str)
        img = Image.open(image_full_path)
        pic_arr = np.array(img)
        mean_channel_dataset += (pic_arr / 225.).mean(axis=(0, 1))
    mean_channel_dataset /= pics_num

    # calc std
    for i, face_pic_str in enumerate(faces_file_strs):
        image_full_path = os.path.join(dataset_path, face_pic_str)
        img = Image.open(image_full_path)
        pic_arr = np.array(img)
        std_channel_dataset += ((pic_arr / 225.) ** 2).mean(axis=(0, 1)) - mean_channel_dataset ** 2
    std_channel_dataset /= pics_num
    std_channel_dataset = np.sqrt(std_channel_dataset)

    print("Train mean:", mean_channel_dataset, "Train std:", std_channel_dataset)
    return mean_channel_dataset, std_channel_dataset


print("Creating utkface bw balanced dataset:")
create_sets(src_dir, os.path.join(dest_dir, "bw_balanced"), override=False)
