#!/bin/bash

# Network hyperparameters
num_grad_steps=5
num_resblocks=2
num_features=64
device=1

# Name of model
model_name=train-3D_$((num_grad_steps))steps_$((num_resblocks))resblocks_$((num_features))features

# Set folder names
dir_data=/data/sandino/Cine
dir_summary=$dir_data/summary/$model_name

python3 train.py --data-path $dir_data \
				 --exp-dir $dir_summary \
				 --num-grad-steps $num_grad_steps \
				 --num-resblocks $num_resblocks \
				 --num-features $num_features \
				 --num-emaps 2 \
				 --slwin-init \
				 --device-num $device 