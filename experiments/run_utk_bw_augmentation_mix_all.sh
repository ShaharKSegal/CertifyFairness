#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python main.py	 --data_path data/UTKFace/bw_balanced --lr 0.01 --max_epochs=100\
		 --ignore_loss_weights\
		 --transform_train_rotate_prob 0.32 --transform_test_rotate_prob 0.32\
		 --transform_train_crop_prob 0.59 --transform_test_crop_prob 0.59\
		 --transform_train_rectangle_erasing_prob 0.74 --transform_test_rectangle_erasing_prob 0.74\
		 --transform_train_gaussian_noise_std 0.1 --transform_test_gaussian_noise_std 0.1\
		 --train_path data/UTKFace/bw_balanced/all\
		 --eval_path data/UTKFace/bw_balanced/all\
		 --runname utk_bw_all_fair_augment_mix_same_prob --verbose

python main.py	 --data_path data/UTKFace/bw_balanced --lr 0.01 --max_epochs=100\
		 --group_weights 1.0 1.0 1.0 0.01 --ignore_loss_weights\
		 --transform_train_rotate_prob 0.32 --transform_test_rotate_prob 0.32\
		 --transform_train_crop_prob 0.59 --transform_test_crop_prob 0.59\
		 --transform_train_rectangle_erasing_prob 0.74 --transform_test_rectangle_erasing_prob 0.74\
		 --transform_train_gaussian_noise_std 0.1 --transform_test_gaussian_noise_std 0.1\
		 --train_path data/UTKFace/bw_balanced/all\
		 --eval_path data/UTKFace/bw_balanced/all\
		 --runname utk_bw_all_unfair_augment_mix_same_prob --verbose

python main.py	 --data_path data/UTKFace/bw_balanced --lr 0.01 --max_epochs=100\
		 --ignore_loss_weights\
		 --transform_train_rotate_prob 0.32 --transform_test_rotate_prob 0.56\
		 --transform_train_crop_prob 0.59 --transform_test_crop_prob 0.23\
		 --transform_train_rectangle_erasing_prob 0.74 --transform_test_rectangle_erasing_prob 0.24\
		 --transform_train_gaussian_noise_std 0.1 --transform_test_gaussian_noise_std 0.04\
		 --train_path data/UTKFace/bw_balanced/all\
		 --eval_path data/UTKFace/bw_balanced/all\
		 --runname utk_bw_all_fair_augment_mix_diff_prob --verbose

python main.py	 --data_path data/UTKFace/bw_balanced --lr 0.01 --max_epochs=100\
		 --group_weights 1.0 1.0 1.0 0.01 --ignore_loss_weights\
		 --transform_test_rotate_prob 0.32 --transform_train_rotate_prob 0.56\
		 --transform_test_crop_prob 0.59 --transform_train_crop_prob 0.23\
		 --transform_train_rectangle_erasing_prob 0.74 --transform_test_rectangle_erasing_prob 0.24\
		 --transform_train_gaussian_noise_std 0.1 --transform_test_gaussian_noise_std 0.04\
		 --train_path data/UTKFace/bw_balanced/all\
		 --eval_path data/UTKFace/bw_balanced/all\
		 --runname utk_bw_all_unfair_augment_mix_diff_prob --verbose
