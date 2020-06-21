#!/bin/bash

case="result_20200620_artfield_trajgru_size200_lr0002"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_trajGRU_jma.py --model_name trajgru \
       --dataset artfield --model_mode run --data_scaling linear\
       --data_path ../data/artificial/uniform/ --image_size 200 \
       --valid_data_path ../data/artificial/uniform/ \
       --train_path ../data/artificial_uniform_train.csv \
       --valid_path ../data/artificial_uniform_valid.csv \
       --test --eval_threshold 0.5 10 20 --test_path ../data/artificial_uniform_valid.csv \
       --result_path $case --tdim_use 12 --learning_rate 0.0002 --lr_decay 0.9 \
       --batch_size 10 --n_epochs 20 --n_threads 4 --checkpoint 10 \
       --hidden_channels 5 --kernel_size 3 --optimizer adam
# png output
# python ../src_post/plot_comp_prediction_trajgru.py $case
# python ../src_post/gif_animation.py $case
