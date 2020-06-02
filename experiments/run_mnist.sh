#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python main.py	 --dataset mnist2 --data_path data/MNIST/colored --model lenet --lr 0.01 --max_epochs=25\
		 --ignore_loss_weights\
		 --runname mnist_fair --verbose

python main.py	 --dataset mnist2 --data_path data/MNIST/colored --model lenet --lr 0.01 --max_epochs=25\
		 --group_weights 1.0 1.0 1.0 0.0 --ignore_loss_weights\
		 --runname mnist_unfair_1110 --verbose

python main.py	 --dataset mnist2 --data_path data/MNIST/colored --model lenet --lr 0.01 --max_epochs=25\
		 --group_weights 0.0 1.0 1.0 0.0 --ignore_loss_weights\
		 --runname mnist_unfair_0110 --verbose
