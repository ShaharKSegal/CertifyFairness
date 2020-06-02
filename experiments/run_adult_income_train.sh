#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3
python main.py   --data_path data/adult_income --dataset adult_income --model simple_nn \
		 --lr 0.01 --max_epochs=50 --ignore_loss_weights \
                 --runname adult_income_fair_sample --verbose
