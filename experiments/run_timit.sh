#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3
python main.py	 --dataset timit2 --model lenet --data_path data/timit --lr 0.005 --max_epochs=50\
		 --ignore_sampling_weights\
		 --runname timit_row_count_100 --verbose
python main.py	 --dataset timit2 --model lenet --data_path data/timit --lr 0.005 --max_epochs=50\
		 --ignore_weights\
		 --runname timit_no_weights_row_count_100 --verbose
