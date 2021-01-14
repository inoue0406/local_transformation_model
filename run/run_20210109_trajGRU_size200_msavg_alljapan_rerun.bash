#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

case="result_20210109_trajGRU_size200_alljapan_msavg_lr0002_rerun"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_trajGRU_jma.py --model_name trajgru --no_train\
       --dataset radarJMA_msavg --model_mode run --data_scaling linear\
       --data_path ../data/data_alljapan/\
       --avg_path ../data/data_alljapan_multiavg/ --image_size 200\
       --valid_data_path ../data/data_kanto/ \
       --train_path ../data/train_alljapan_2yrs_JMARadar_msavg.csv \
       --valid_path ../data/valid_simple_JMARadar.csv\
       --transfer_path ../run/result_20210103_trajGRU_size200_alljapan_msavg_lr0002/trained_CLSTM.dict\
       --test --eval_threshold 20\
       --test_path ../data/valid_simple_JMARadar_msavg.csv\
       --test_avg_path ../data/data_kanto_multiavg/\
       --result_path $case --tdim_use 12 --tdim_loss 12 --learning_rate 0.0002 --lr_decay 0.9\
       --num_input_layer 11\
       --batch_size 20 --n_epochs 20 --n_threads 10 --checkpoint 10 \
       --loss_function MSE\
       --optimizer adam \
# post plot
#python ../src_post/plot_comp_prediction_trajgru.py $case
#python ../src_post/gif_animation.py $case

