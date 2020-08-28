#!/bin/bash

case="result_20200825_artfield_traj_el_faiss_size200_pc100_12step_lr0001"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_trajGRU_jma.py --model_name trajgru_el \
       --dataset artfield --model_mode run --data_scaling linear\
       --data_path ../data/artificial/nouniform/ --image_size 200 --pc_size 100\
       --aug_rotate 0 --aug_resize 0\
       --valid_data_path ../data/artificial/nouniform/ \
       --train_path ../data/artificial_uniform_train.csv \
       --valid_path ../data/artificial_uniform_valid.csv \
       --test --eval_threshold 0.5 10 20 --test_path ../data/artificial_uniform_valid.csv \
       --result_path $case --tdim_use 12 --tdim_loss 12 --learning_rate 0.001 --lr_decay 0.8 \
       --batch_size 10 --n_epochs 20 --n_threads 4 --checkpoint 10 \
       --loss_function MSE --interp_type nearest\
       --hidden_channels 5 --kernel_size 3 --optimizer adam
