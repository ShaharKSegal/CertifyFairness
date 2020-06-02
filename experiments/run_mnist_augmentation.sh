#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python main.py	 --dataset mnist2 --data_path data/MNIST/colored --model lenet --lr 0.01 --max_epochs=50\
		 --ignore_loss_weights\
		 --transform_test_rotate_prob 0.32 --transform_train_rotate_prob 0.56\
		 --transform_test_crop_prob 0.59 --transform_train_crop_prob 0.23\
		 --transform_train_crop_size 20 --transform_test_crop_size 20\
		 --transform_train_rectangle_erasing_prob 0.74 --transform_test_rectangle_erasing_prob 0.24\
		 --transform_train_gaussian_noise_std 0.05 --transform_test_gaussian_noise_std 0.02\
		 --eval_path data/MNIST/colored/train\
		 --runname mnist_fair_augmentation_mix --verbose

python main.py	 --dataset mnist2 --data_path data/MNIST/colored --model lenet --lr 0.01 --max_epochs=50\
		 --group_weights 1.0 1.0 1.0 0.0 --ignore_loss_weights\
		 --transform_test_rotate_prob 0.32 --transform_train_rotate_prob 0.56\
		 --transform_test_crop_prob 0.59 --transform_train_crop_prob 0.23\
		 --transform_train_crop_size 20 --transform_test_crop_size 20\
		 --transform_train_rectangle_erasing_prob 0.74 --transform_test_rectangle_erasing_prob 0.24\
		 --transform_train_gaussian_noise_std 0.05 --transform_test_gaussian_noise_std 0.02\
		 --eval_path data/MNIST/colored/train\
		 --runname mnist_unfair_1110_augmentation_mix --verbose

python main.py	 --dataset mnist2 --data_path data/MNIST/colored --model lenet --lr 0.01 --max_epochs=50\
		 --group_weights 0.0 1.0 1.0 0.0 --ignore_loss_weights\
		 --transform_test_rotate_prob 0.32 --transform_train_rotate_prob 0.56\
		 --transform_test_crop_prob 0.59 --transform_train_crop_prob 0.23\
		 --transform_train_crop_size 20 --transform_test_crop_size 20\
		 --transform_train_rectangle_erasing_prob 0.74 --transform_test_rectangle_erasing_prob 0.24\
		 --transform_train_gaussian_noise_std 0.05 --transform_test_gaussian_noise_std 0.02\
		 --eval_path data/MNIST/colored/train\
		 --runname mnist_unfair_0110_augmentation_mix --verbose
