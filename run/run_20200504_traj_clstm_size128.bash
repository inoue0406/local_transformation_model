#!/bin/bash

case="result_20200504_traj_clstm_size128"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_trajGRU_jma.py --model_name clstm --data_path ../data/data_kanto_resize/ \
       --valid_data_path ../data/data_kanto_resize/ \
       --train_path ../data/train_simple_JMARadar.csv \
       --valid_path ../data/valid_simple_JMARadar.csv \
       --test --eval_threshold 0.5 10 20 --test_path ../data/valid_simple_JMARadar.csv \
       --result_path $case --tdim_use 12 --learning_rate 0.02 --lr_decay 0.7 \
       --batch_size 10 --n_epochs 20 --n_threads 4 --checkpoint 10 \
       --hidden_channels 8 --kernel_size 3 --optimizer adam
# png output
python ../src_post/plot_comp_prediction.py $case
python ../src_post/gif_animation.py $case
