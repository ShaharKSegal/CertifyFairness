#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python main.py	 --dataset celeba2 --data_path data/celeba --lr 0.01 --max_epochs=20\
		 --group_weights 1.0 1.0 1.0 0.01 --ignore_loss_weights\
		 --runname celeba_001_reweight --verbose

python main.py	 --dataset celeba2 --data_path data/celeba --lr 0.01 --max_epochs=20\
		 --group_weights 1.0 1.0 1.0 0.001 --ignore_loss_weights\
		 --runname celeba_0001_reweight --verbose

python main.py	 --dataset celeba2 --data_path data/celeba --lr 0.01 --max_epochs=20\
		 --group_weights 1.0 1.0 1.0 0.0001 --ignore_loss_weights\
		 --runname celeba_00001_reweight --verbose
