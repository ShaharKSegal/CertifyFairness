#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python main.py	 --data_path data/UTKFace/bw_balanced --lr 0.01 --max_epochs=50\
		 --group_weights 1.0 1.0 1.0 0.1 --ignore_loss_weights\
		 --runname utk_bw_black_samp_weights_1_01 --verbose
python main.py	 --data_path data/UTKFace/bw_balanced --lr 0.01 --max_epochs=50\
		 --group_weights 1.0 1.0 1.0 0.5 --ignore_loss_weights\
		 --runname utk_bw_black_samp_weights_1_05 --verbose
python main.py	 --data_path data/UTKFace/bw_balanced --lr 0.01 --max_epochs=50\
		 --group_weights 1.0 1.0 1.0 0.05 --ignore_loss_weights\
		 --runname utk_bw_black_samp_weights_1_005 --verbose

python main.py	 --data_path data/UTKFace/bw_balanced --lr 0.01 --max_epochs=50\
		 --group_weights 1.0 10.0 1.0 1.0 --ignore_loss_weights\
		 --runname utk_bw_black_samp_weights_10_1 --verbose
python main.py	 --data_path data/UTKFace/bw_balanced --lr 0.01 --max_epochs=50\
		 --group_weights 1.0 10.0 1.0 0.1 --ignore_loss_weights\
		 --runname utk_bw_black_samp_weights_10_01 --verbose
python main.py	 --data_path data/UTKFace/bw_balanced --lr 0.01 --max_epochs=50\
		 --group_weights 1.0 10.0 1.0 0.01 --ignore_loss_weights\
		 --runname utk_bw_black_samp_weights_10_001 --verbose

