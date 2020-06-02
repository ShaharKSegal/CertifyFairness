#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python main.py	 --dataset lfw --data_path data/lfw --lr 0.001 --max_epochs=50\
		 --ignore_loss_weights\
		 --runname lfw_samp_weights_lr_001 --verbose
