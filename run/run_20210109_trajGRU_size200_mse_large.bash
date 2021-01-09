#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

case="result_20210109_trajGRU_size200_mse_lr0002_large"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_trajGRU_jma.py --model_name trajgru\
       --dataset radarJMA --model_mode run --data_scaling linear\
       --data_path ../data/data_kanto/ --image_size 200\
       --valid_data_path ../data/data_kanto/ \
       --train_path ../data/train_simple_JMARadar.csv \
       --valid_path ../data/valid_simple_JMARadar.csv \
       --test --eval_threshold 0.5 10 20 --test_path ../data/valid_simple_JMARadar.csv \
       --result_path $case --tdim_use 12 --tdim_loss 12 --learning_rate 0.0002 --lr_decay 0.9\
       --batch_size 8 --n_epochs 20 --n_threads 4 --checkpoint 10 \
       --loss_function MSE\
       --num_filters 8 64 192 192\
       --optimizer adam \
# post plot
python ../src_post/plot_comp_prediction_trajgru.py $case
python ../src_post/gif_animation.py $case
