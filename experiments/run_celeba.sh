#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python main.py	 --dataset celeba2 --data_path data/celeba --lr 0.01 --max_epochs=20\
		 --ignore_loss_weights\
		 --runname celeba_fair --verbose
