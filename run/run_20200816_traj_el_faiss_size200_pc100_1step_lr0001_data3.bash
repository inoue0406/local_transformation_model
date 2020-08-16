#!/bin/bash

case="result_20200816_traj_el_faiss_size200_pc100_1step_lr0001_data3"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_trajGRU_jma.py --model_name trajgru_el \
       --dataset radarJMA3 --model_mode run --data_scaling linear\
       --data_path ../data/data_kanto_aug/ --image_size 200 --pc_size 100\
       --valid_data_path ../data/data_kanto_aug/ \
       --train_path ../data/filelist_triplet_train_JMARadar_nozero.csv \
       --valid_path ../data/filelist_triplet_valid_JMARadar_nozero.csv \
       --test --eval_threshold 0.5 10 20 --test_path ../data/filelist_triplet_valid_JMARadar_nozero.csv \
       --result_path $case --tdim_use 12 --tdim_loss 1 --learning_rate 0.0001 --lr_decay 0.8 \
       --batch_size 10 --n_epochs 4 --n_threads 4 --checkpoint 10 \
       --loss_function MSE --interp_type nearest\
       --hidden_channels 5 --kernel_size 3 --optimizer adam
#       --transfer_path  ../run/result_20200615_artfield_value_traj_el_size200/trained_CLSTM.model \
