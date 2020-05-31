import argparse
import dataclasses

import torch

from typing import List


# todo: have ArgsTransformation take over the relevant arguments for train and test to make access less of a pain
@dataclasses.dataclass
class ArgsTransformation:
    rectangle_erasing_prob: float
    gaussian_noise_std: float

    rotate_angle: int
    rotate_prob: float

    crop_size: int
    crop_prob: float


class ArgsNamespace(argparse.Namespace):
    task: str
    dataset: str
    model: str
    optimizer: str

    group_weights: List[float]
    ignore_weights: bool
    ignore_loss_weights: bool
    ignore_sampling_weights: bool

    # transformation arguments
    # transform_random_order: bool
    transform_train_rectangle_erasing_prob: float
    transform_test_rectangle_erasing_prob: float

    transform_train_gaussian_noise_std: float
    transform_test_gaussian_noise_std: float

    transform_train_rotate_angle: int
    transform_train_rotate_prob: float
    transform_test_rotate_angle: int
    transform_test_rotate_prob: float

    transform_train_crop_size: int
    transform_train_crop_prob: float
    transform_test_crop_size: int
    transform_test_crop_prob: float

    mislabel: List[float]

    knn_k: int

    data_path: str
    train_path: str
    eval_path: str
    test_path: str

    ignore_cuda: bool
    lr: float
    batch_size: int
    max_epochs: int

    runname: str
    save_dir: str
    summary_file: str
    save_model: str
    save_stats: str
    ignore_timestamp: bool

    verbose: bool

    def __init__(self, **kwargs):
        super().__init__()

    @property
    def torch_device(self):
        return 'cuda' if torch.cuda.is_available() and not self.ignore_cuda else 'cpu'

    @property
    def use_cuda(self):
        return self.torch_device == 'cuda'
