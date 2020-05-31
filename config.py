import argparse
import os
import logging

import parse

log: logging.Logger
args: parse.ArgsNamespace

train_log_interval = 10

default_data_path = os.path.join(".", "data")

UTKFace_BW = 'utkface_bw'
UTKFace_BW_balanced = 'utkface_bw_balanced'
UTKFace_Full = 'utkface_full'
UTKFace_lst = [UTKFace_BW, UTKFace_Full, UTKFace_BW_balanced]
AdultIncome = 'adult_income'
LFW = 'lfw'
SyntheticData = "synthetic"
Timit = 'timit'
Timit2Groups = 'timit2'
Timit_lst = [Timit, Timit2Groups]
CelebA = 'celeba'
CelebA2Groups = 'celeba2'
CelebA_lst = [CelebA, CelebA2Groups]
ColoredMNIST = 'mnist'
ColoredMNIST2Groups = 'mnist2'
ColoredMNIST_lst = [ColoredMNIST, ColoredMNIST2Groups]

datasets = [*UTKFace_lst, AdultIncome, SyntheticData, LFW, *Timit_lst, *CelebA_lst, *ColoredMNIST_lst]

fairness_training = 'fair'
augment_testing_regular = 'augment_test'
augment_testing_knn = 'augment_test_knn'
augment_testing_odin = "augment_test_odin"
sklearn_regular_training = 'sklearn'
evaluate_fairness = "eval_fair"
augment_testing_lst = [augment_testing_regular, augment_testing_knn, augment_testing_odin]
tasks = [fairness_training, sklearn_regular_training, evaluate_fairness, *augment_testing_lst]

Resnet18 = 'resnet18'
OverfitResnet = 'overfit_resnet'
MLP = "mlp"
LeNet = 'lenet'
RandomForest = 'random_forest'
LinearSVM = 'linear_svm'
SVM = 'svm'
torch_models = [Resnet18, MLP, LeNet, OverfitResnet]
sklearn_models = [RandomForest, LinearSVM, SVM]
models = [*sklearn_models, *torch_models]

Adam = 'adam'
SGD = 'sgd'
optimizers = [Adam, SGD]
