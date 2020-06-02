#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python main.py	 --task eval_fair --dataset timit2 --model lenet --data_path data/timit --lr 0.005 --max_epochs=50\
		 --ignore_sampling_weights\
		 --runname timit_no_weights_lr_005_20191110_162404 --ignore_timestamp --verbose
