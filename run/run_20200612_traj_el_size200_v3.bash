#!/bin/bash

case="result_20200612_traj_el_size200_v3"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_trajGRU_jma.py --model_name trajgru_el \
       --data_path ../data/data_kanto/ --image_size 200\
       --valid_data_path ../data/data_kanto/ \
       --train_path ../data/train_kanto_flatsampled_JMARadar.csv \
       --valid_path ../data/valid_simple_JMARadar.csv \
       --test --eval_threshold 0.5 10 20 --test_path ../data/valid_simple_JMARadar.csv \
       --result_path $case --tdim_use 12 --learning_rate 0.002 --lr_decay 0.9 \
       --batch_size 10 --n_epochs 50 --n_threads 4 --checkpoint 10 \
       --loss_function MSE \
       --hidden_channels 5 --kernel_size 3 --optimizer adam
# png output
python ../src_post/plot_comp_prediction_trajgru.py $case
python ../src_post/gif_animation.py $case

