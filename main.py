import argparse
import datetime
import os

import config
import dataset
import parse
import logger
import task

parser = argparse.ArgumentParser(description='Train Fairness Model')

parser.add_argument('--task', default=config.fairness_training, choices=config.tasks, help='the task to perform')
parser.add_argument('--dataset', default=config.UTKFace_BW_balanced, choices=config.datasets,
                    help='the dataset to train on')
parser.add_argument('--model', default=config.Resnet18, choices=config.models, help='the model to train')
parser.add_argument('--optimizer', default=config.Adam, choices=config.optimizers, help='the optimizer for training')

# groups weights arguments
parser.add_argument('--group_weights', default=[0.0], type=float, nargs='*',
                    help='adds group weights, by default the groups are balanced for each label+fairness group. '
                         'either by a list of decimals with length = (#label)*(#fairness groups) '
                         'or a single value, which adds the bias towards the majority')
parser.add_argument('--ignore_weights', action='store_true',
                    help='force groups to have equal weights (unbalanced), ignores the group_weights option')
parser.add_argument('--ignore_loss_weights', action='store_true',
                    help='force the loss to ignore the group weights, ignores the group_weights option for loss')
parser.add_argument('--ignore_sampling_weights', action='store_true',
                    help='force data to be sampled uniformly, ignores the group_weights option for sampling')

# transformation image arguments
parser.add_argument('--activate_transformations', action='store_true',
                    help='invokes the custom transformation')
parser.add_argument('--transform_train_rectangle_erasing_prob', default=0, type=float,
                    help='adds image rectangle erasing to training transformation. '
                         'Relevant only for image datasets, ignored otherwise')
parser.add_argument('--transform_test_rectangle_erasing_prob', default=0, type=float,
                    help='adds image rectangle erasing to training transformation. '
                         'Relevant only for image datasets, ignored otherwise')
parser.add_argument('--transform_train_gaussian_noise_std', default=0, type=float,
                    help='adds image rectangle erasing to training transformation. '
                         'Relevant only for image datasets, ignored otherwise')
parser.add_argument('--transform_test_gaussian_noise_std', default=0, type=float,
                    help='adds image rectangle erasing to training transformation. '
                         'Relevant only for image datasets, ignored otherwise')
parser.add_argument('--transform_train_rotate_prob', default=0, type=float,
                    help='adds image rotation probability to training transformation. '
                         'Relevant only for image datasets, ignored otherwise')
parser.add_argument('--transform_test_rotate_prob', default=0, type=float,
                    help='adds image rotation probability to test transformation. '
                         'Relevant only for image datasets, ignored otherwise')
parser.add_argument('--transform_train_crop_prob', default=0, type=float,
                    help='adds image crop probability to training transformation. '
                         'Relevant only for image datasets, ignored otherwise')
parser.add_argument('--transform_test_crop_prob', default=0, type=float,
                    help='adds image crop probability to test transformation. '
                         'Relevant only for image datasets, ignored otherwise')

parser.add_argument('--transform_train_rotate_angle', default=0, type=int,
                    help='adds image rotation max angle to training transformation. '
                         'Relevant only for image datasets, ignored otherwise')
parser.add_argument('--transform_test_rotate_angle', default=0, type=int,
                    help='adds image rotation max angle to test transformation. '
                         'Relevant only for image datasets, ignored otherwise')

parser.add_argument('--transform_train_crop_size', default=1000, type=int,
                    help='adds image crop to training transformation. '
                         'Relevant only for image datasets, ignored otherwise')
parser.add_argument('--transform_test_crop_size', default=1000, type=int,
                    help='adds image crop to test transformation. '
                         'Relevant only for image datasets, ignored otherwise')

# mislabeling
parser.add_argument('--mislabel', default=[0.0], type=float, nargs='*',
                    help='adds mislabeling, by default the labels are correct for each label+fairness group. '
                         'either by a list of decimals between 0.0 to 1.0 with length = (#label)*(#fairness groups) '
                         'or a single value, which adds mislabeling on the smallest minority')

# k-nn arguments
parser.add_argument('--knn_k', type=int, default=1, help='k nearest neighbours k hyperparameter')

# dataset path arguments
parser.add_argument('--data_path', default=config.default_data_path,
                    help='the dataset root path, used to find default train, eval and test paths')
parser.add_argument('--train_path', help='the path to the dir of the training data, overrides data_path')
parser.add_argument('--eval_path', help='the path to the dir of the evaluation data, overrides data_path')
parser.add_argument('--test_path', help='the path to the dir of the testing data, overrides data_path')

# learning arguments
parser.add_argument('--ignore_cuda', action='store_true', help='should ignore cuda on device (only for DNN)')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch_size', default=100, type=int, help='the batch size')
parser.add_argument('--max_epochs', default=30, type=int, help='the maximum number of epochs')

# experiment arguments
parser.add_argument('--runname', default='train', help='the experiment name')
parser.add_argument('--save_dir', default='./saved_runs/', help='the path to the root run dir')
parser.add_argument('--summary_file', default='runs_summary.csv', help='the summary file name. documents each exp')
parser.add_argument('--save_model', default='model.model', help='model file name')
parser.add_argument('--save_stats', default='stats.pkl', help='statistics file name')
parser.add_argument('--ignore_timestamp', action='store_true', help='dont add datetime stamp to run dir')

parser.add_argument('--verbose', action='store_true', help='print info to screen')

config.args = parser.parse_args(namespace=parse.ArgsNamespace())

# create run dir and setup
if config.args.ignore_weights:
    config.args.ignore_sampling_weights = config.args.ignore_loss_weights = True
if not config.args.ignore_timestamp:
    config.args.runname += datetime.datetime.now().strftime('_%Y%m%d_%H%M%S')
run_dir = task_dir = os.path.join(config.args.save_dir, config.args.runname)

if config.args.task in config.augment_testing_lst:
    task_dir = os.path.join(task_dir, config.args.task, config.args.runname,
                            datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

os.makedirs(task_dir, exist_ok=True)
logger.set_logger(os.path.join(task_dir, 'log_' + str(config.args.runname) + '.log'))
configfile = os.path.join(task_dir, 'conf_' + str(config.args.runname) + '.config')

config.log.info(f'==> Created subdir for run at: {task_dir}')

# args validation
if config.args.task == config.fairness_training:
    if not config.args.ignore_weights and not config.args.ignore_sampling_weights \
            and not config.args.ignore_loss_weights:
        config.log.warning("Both sampling and loss uses weights, it is advised to ignore weights for either one")
if config.args.task == config.augment_testing_odin:
    config.args.batch_size = 1

# save configuration parameters
with open(configfile, 'w') as f:
    for arg in vars(config.args):
        f.write('{}: {}\n'.format(arg, getattr(config.args, arg)))

config.log.info(f'Running task {config.args.task}...')
config.log.info('==> Loading dataset...')
train_loader, eval_loader, test_loader = dataset.get_loaders()

# run the task, usually training a model
task.run_task(run_dir, task_dir, train_loader, eval_loader, test_loader)
