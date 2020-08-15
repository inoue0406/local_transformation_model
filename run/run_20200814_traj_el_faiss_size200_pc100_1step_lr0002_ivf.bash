#!/bin/bash

case="result_20200814_traj_el_faiss_size200_pc100_1step_lrd09_ivf"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_trajGRU_jma.py --model_name trajgru_el \
       --dataset radarJMA --model_mode run --data_scaling linear\
       --data_path ../data/data_kanto/ --image_size 200 --pc_size 100\
       --valid_data_path ../data/data_kanto/ \
       --train_path ../data/train_simple_JMARadar.csv \
       --valid_path ../data/valid_simple_JMARadar.csv \
       --test --eval_threshold 0.5 10 20 --test_path ../data/valid_simple_JMARadar.csv \
       --result_path $case --tdim_use 12 --tdim_loss 1 --learning_rate 0.0001 --lr_decay 0.9 \
       --batch_size 10 --n_epochs 20 --n_threads 4 --checkpoint 10 \
       --loss_function MSE --interp_type nearest\
       --hidden_channels 5 --kernel_size 3 --optimizer adam

case="result_20200814_traj_el_faiss_size200_pc100_12step_lrd09_trans"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_trajGRU_jma.py --model_name trajgru_el --no_train \
       --dataset radarJMA --model_mode run --data_scaling linear\
       --data_path ../data/data_kanto/ --image_size 200 --pc_size 100\
       --valid_data_path ../data/data_kanto/ \
       --train_path ../data/train_simple_JMARadar.csv \
       --valid_path ../data/valid_simple_JMARadar.csv \
       --test --eval_threshold 0.5 10 20 --test_path ../data/valid_simple_JMARadar.csv \
       --result_path $case --tdim_use 12 --tdim_loss 12 --learning_rate 0.0001 --lr_decay 0.9 \
       --batch_size 10 --n_epochs 20 --n_threads 4 --checkpoint 10 \
       --loss_function MSE --interp_type nearest\
       --hidden_channels 5 --kernel_size 3 --optimizer adam \
       --transfer_path  ../run/result_20200814_traj_el_faiss_size200_pc100_1step_lrd09_ivf/trained_CLSTM.dict \
