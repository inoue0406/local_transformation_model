#!/bin/bash

case="result_20200830_traj_el_hybrid_size200_pc200_1step_lr0001_msloss"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_trajGRU_jma.py --model_name trajgru_el \
       --dataset radarJMA --model_mode run --data_scaling linear\
       --data_path ../data/data_kanto/ --image_size 200 --pc_size 200\
       --aug_rotate 0 --aug_resize 0\
       --valid_data_path ../data/data_kanto/ \
       --train_path ../data/train_simple_JMARadar.csv \
       --valid_path ../data/valid_simple_JMARadar.csv \
       --test --eval_threshold 0.5 10 20 --test_path ../data/valid_simple_JMARadar.csv \
       --result_path $case --tdim_use 12 --tdim_loss 1 --learning_rate 0.00002 --lr_decay 0.9 \
       --batch_size 10 --n_epochs 20 --n_threads 4 --checkpoint 10 \
       --loss_function MultiMSE --loss_weights 1.0 0.05 0.0125 0.0025\
       --interp_type nearest\
       --hidden_channels 5 --kernel_size 3 --optimizer adam

case="result_20200830_traj_el_hybrid_size200_pc200_12step_lr0001_msloss"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_trajGRU_jma.py --model_name trajgru_el\
       --dataset radarJMA --model_mode run --data_scaling linear\
       --data_path ../data/data_kanto/ --image_size 200 --pc_size 100\
       --aug_rotate 0 --aug_resize 0\
       --valid_data_path ../data/data_kanto/ \
       --train_path ../data/train_simple_JMARadar.csv \
       --valid_path ../data/valid_simple_JMARadar.csv \
       --test --eval_threshold 0.5 10 20 --test_path ../data/valid_simple_JMARadar.csv \
       --result_path $case --tdim_use 12 --tdim_loss 12 --learning_rate 0.00002 --lr_decay 0.9 \
       --batch_size 10 --n_epochs 20 --n_threads 4 --checkpoint 10 \
       --loss_function MultiMSE --loss_weights 1.0 0.05 0.0125 0.0025\
       --interp_type nearest\
       --hidden_channels 5 --kernel_size 3 --optimizer adam \
       --transfer_path  ../run/result_20200830_traj_el_hybrid_size200_pc200_1step_lr0001_msloss/trained_CLSTM.dict \
