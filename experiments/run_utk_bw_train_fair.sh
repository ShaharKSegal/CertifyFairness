#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python main.py	 --data_path data/UTKFace/bw_balanced --lr 0.01 --max_epochs=50\
		 --ignore_sampling_weights\
		 --runname utk_bw_fair_loss --verbose

python main.py	 --data_path data/UTKFace/bw_balanced --lr 0.01 --max_epochs=50\
		 --ignore_weights\
		 --runname utk_bw_no_weights --verbose

python main.py	 --data_path data/UTKFace/bw_balanced --lr 0.01 --max_epochs=50\
		 --ignore_loss_weights\
		 --runname utk_bw_fair_sampling --verbose

