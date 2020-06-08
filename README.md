
# Fairness in the Eyes of the Data

This repository is the official implementation of the experiments in Fairness in the Eyes of the Data (link when published). 
Scripts we used to run the experiments with are available at [experiments](https://github.com/ShaharKSegal/CertifyFairness/tree/master/experiments).

## Requirements

Runs on python 3.7+ install requirements:

```setup
pip install -r requirements.txt
```

## Training

For regular training, run the following command:

```regular train
python main.py --dataset <dataset_name> --data_path <path_to_dataset> --model <model> --lr <lr> --max_epochs=<n> --ignore_loss_weights --runname <name_your_run> --verbose
```
For bias training, adjust the weights argument as suits your need:

```bias train
python main.py --dataset <dataset_name> --data_path <path_to_dataset> --model <model> --lr <lr> --max_epochs=<n> --ignore_loss_weights --runname <name_your_run> --ignore_timestamp --verbose --group_weights <bias weights>
```

For augmented training:
```augmented train
python main.py --dataset <dataset_name> --data_path <path_to_dataset> --model <model> --lr <lr> --max_epochs=<n> --ignore_loss_weights --runname <name_your_run> --verbose \
     --activate_transformations\
		 --transform_test_rotate_prob <p1> --transform_train_rotate_prob <p2>\
		 --transform_test_crop_prob <p3> --transform_train_crop_prob <p4>\
		 --transform_train_crop_size <p5> --transform_test_crop_size <p6>\
		 --transform_train_rectangle_erasing_prob <p7> --transform_test_rectangle_erasing_prob <p8>\
		 --transform_train_gaussian_noise_std <p9> --transform_test_gaussian_noise_std <p10>\
     --transform_train_rotate_angle <m1> --transform_test_rotate_angle <m2>\
     --transform_train_crop_size <m3> --transform_test_crop_size <m4>
```

Documentation of each parameter and other options are available via running:
```help
python main.py --help
```

## Evaluation

For evaluation of likelihood and risk, use the same parameters for training with the name of the run, model and dataset:
```fair eval
python main.py --task fair_eval --dataset <dataset_name> --data_path <path_to_dataset> --model <model> --lr <lr> --max_epochs=<n> --ignore_loss_weights --runname <name_of_run> --ignore_timestamp --verbose --group_weights <bias weights>
```

For evaluation of augmentation, use the same parameters for training with the name of the run, model and dataset:
```fair eval
python main.py --task augment_testing_regular --dataset <dataset_name> --data_path <path_to_dataset> --model <model> --lr <lr> --max_epochs=<n> --ignore_loss_weights --runname <name_of_run> --ignore_timestamp --verbose --group_weights <bias weights>
```

## Results

We trained the models in regular or bias mode and tested the empirical fairness gap in the private setup:

### [UTKFace](https://susanqq.github.io/UTKFace/), [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [Colored-MNIST (section 5.2)](https://arxiv.org/abs/1907.02893)

| Dataset         | Model         |  Accuracy  | Empirical Fairness Gap |
| --------------- | ------------- | ---------- | ---------------------- |
| UTKFace         | ResNet18      |   89.76%   |      0.012             |
| UTKFace         | Bias-ResNet18 |   88.56%   |      0.093             |
| CelebA          | ResNet18      |   97.63%   |      0.007             |
| CelebA          | Bias-ResNet18 |   96.95%   |      0.034             |
| C-MNIST         | LeNet         |   98.11%   |      0.001             |
| C-MNIST         | Bias-LeNet    |   74.01%   |      0.450             |

In the public augmented setup:

| Dataset         | Model         |  Accuracy  | Empirical Fairness Gap |
| --------------- | ------------- | ---------- | ---------------------- |
| UTKFace         | ResNet18      |   96.11%   |      0.027             |
| UTKFace         | Bias-ResNet18 |   91.44%   |      0.139             |
| CelebA          | ResNet18      |   96.60%   |      0.010             |
| CelebA          | Bias-ResNet18 |   97.02%   |      0.033             |
| C-MNIST         | LeNet         |   89.17%   |      0.007             |
| C-MNIST         | Bias-LeNet    |   67.97%   |      0.340             |
