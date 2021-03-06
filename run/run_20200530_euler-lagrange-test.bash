#!/bin/bash

case="result_20200530_euler-lagrange_lr002"
# 
python ../src/main_Euler-Lagrange_jma.py --data_path ../data/data_kanto_resize/ \
       --valid_data_path ../data/data_kanto_resize/ \
       --train_path ../data/train_simple_JMARadar.csv \
       --valid_path ../data/valid_simple_JMARadar.csv \
       --test --eval_threshold 0.5 10 20 --test_path ../data/valid_simple_JMARadar.csv \
       --result_path $case \
       --model_name euler-lagrange --tdim_use 12 \
       --learning_rate 0.02 --lr_decay 0.9 --image_size 128\
       --batch_size 20 --n_epochs 20 --n_threads 4 --checkpoint 10 \
       --hidden_channels 8 --kernel_size 3 --optimizer adam
# post plot
python ../src_post/plot_comp_prediction_euler-lagrange.py $case
python ../src_post/gif_animation.py $case

case="result_20200530_euler-lagrange_lr0005"
# 
python ../src/main_Euler-Lagrange_jma.py --data_path ../data/data_kanto_resize/ \
       --valid_data_path ../data/data_kanto_resize/ \
       --train_path ../data/train_simple_JMARadar.csv \
       --valid_path ../data/valid_simple_JMARadar.csv \
       --test --eval_threshold 0.5 10 20 --test_path ../data/valid_simple_JMARadar.csv \
       --result_path $case \
       --model_name euler-lagrange --tdim_use 12 \
       --learning_rate 0.005 --lr_decay 0.9 --image_size 128\
       --batch_size 20 --n_epochs 20 --n_threads 4 --checkpoint 10 \
       --hidden_channels 8 --kernel_size 3 --optimizer adam
# post plot
python ../src_post/plot_comp_prediction_euler-lagrange.py $case
python ../src_post/gif_animation.py $case
