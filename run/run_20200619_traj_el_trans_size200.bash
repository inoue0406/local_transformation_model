#!/bin/bash

case="result_20200619_traj_el_trans_size200"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_trajGRU_jma.py --model_name trajgru_el --no_train\
       --dataset radarJMA --model_mode run --data_scaling linear\
       --data_path ../data/data_kanto/ --image_size 200 \
       --valid_data_path ../data/data_kanto/ \
       --train_path ../data/train_simple_JMARadar.csv \
       --valid_path ../data/valid_simple_JMARadar.csv \
       --transfer_path  ../run/result_20200615_artfield_value_traj_el_size200/trained_CLSTM.model \
       --test --eval_threshold 0.5 10 20 --test_path ../data/valid_simple_JMARadar.csv \
       --result_path $case --tdim_use 12 --learning_rate 0.0002 --lr_decay 0.9 \
       --batch_size 10 --n_epochs 20 --n_threads 4 --checkpoint 10 \
       --loss_function MSE \
       --hidden_channels 5 --kernel_size 3 --optimizer adam
# png output
#python ../src_post/plot_comp_prediction_trajgru.py $case
#python ../src_post/gif_animation.py $case

