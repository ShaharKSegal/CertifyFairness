import os

import pandas as pd

import config

from dataset.datasets import TimitDataset

is_train = True
os.chdir('..')
timit_type = config.Timit2Groups
subdir_str = 'train' if is_train else 'test'
path = os.path.join('data/timit', subdir_str)

ds = TimitDataset(path, timit_type, None, 0.0)