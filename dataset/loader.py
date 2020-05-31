import os

import torch.utils.data

import config
import dataset.transform
import dataset.datasets as ds

_ds_dict = {config.UTKFace_Full: ds.UTKFaceDataset,
            config.UTKFace_BW: ds.UTKFaceDataset,
            config.UTKFace_BW_balanced: ds.UTKFaceDataset,
            config.AdultIncome: ds.AdultIncomeDataset,
            config.SyntheticData: ds.SyntheticDataset,
            config.LFW: ds.LFWDataset,
            config.Timit: ds.TimitDataset,
            config.Timit2Groups: ds.TimitDataset,
            config.CelebA: ds.CelebADataset,
            config.CelebA2Groups: ds.CelebADataset,
            config.ColoredMNIST: ds.ColoredMNISTDataset,
            config.ColoredMNIST2Groups: ds.ColoredMNISTDataset}


def get_loaders():
    """
    Creates and returns up to 3 data loader train, eval and test using the given arguments to the program (using main)
    :param task: define which task loader to call
    :param data_path: root dir for the data
    :param dataset_type: which dataset to create
    :param group_weights: adding sample weights to loader
    :param ignore_sampling_weights: Change WeightedRandomSampler to Random Sampler
    :param mislabel: adding mislabeling to data
    :param batch_size: batch size of loader
    :param train_path: override data_path for train path
    :param eval_path:  override data_path for eval path
    :param test_path:  override data_path for test path
    :return: (torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader) - train, eval, test
    """
    dataset_cls = _ds_dict[config.args.dataset]

    # update paths
    train_path = config.args.train_path if config.args.train_path else os.path.join(config.args.data_path,
                                                                                    dataset_cls.train_default)
    eval_path = config.args.eval_path if config.args.eval_path else os.path.join(config.args.data_path,
                                                                                 dataset_cls.eval_default)
    test_path = config.args.test_path if config.args.test_path else os.path.join(config.args.data_path,
                                                                                 dataset_cls.test_default)

    # get dataset transform
    train_transform = dataset.transform.get_transform(config.args.dataset, train_path, True, False)
    eval_transform = dataset.transform.get_transform(config.args.dataset, train_path, False, True)
    test_transform = dataset.transform.get_transform(config.args.dataset, train_path, False, False)

    return get_fair_loaders(dataset_cls, train_path, eval_path, test_path,
                            train_transform, eval_transform, test_transform)


def get_fair_loaders(dataset_cls, train_path, eval_path, test_path, train_transform, eval_transform, test_transform):
    def create_loader(data_set, train_mode=False):
        # add randomized sampling potentially with weights
        if train_mode and config.args.task not in config.augment_testing_lst:
            if config.args.ignore_sampling_weights:
                sampler = torch.utils.data.RandomSampler(data_set)
            else:
                sampler = torch.utils.data.WeightedRandomSampler(data_set.weights, len(data_set), replacement=True)
        else:
            sampler = torch.utils.data.SequentialSampler(data_set)
        return torch.utils.data.DataLoader(dataset=data_set, sampler=sampler, batch_size=config.args.batch_size)

    # create training dataset
    group_weights = None
    if not config.args.ignore_sampling_weights:
        group_weights = config.args.group_weights
    ds_train = dataset_cls(train_path, config.args.dataset, train_transform, group_weights)

    # add mislabeling
    ds_train.mislabel_data(config.args.mislabel)

    # create loaders
    train_loader = create_loader(ds_train, True)
    eval_loader = create_loader(dataset_cls(eval_path, config.args.dataset, eval_transform, None))
    test_loader = create_loader(dataset_cls(test_path, config.args.dataset, test_transform, None))
    return train_loader, eval_loader, test_loader
