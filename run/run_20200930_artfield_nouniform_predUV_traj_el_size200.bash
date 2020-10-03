#!/bin/bash

case="result_20200930_artfield_nouniform_predUV_traj_el_size200"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_trajGRU_jma.py --model_name trajgru_el \
       --dataset artfield --model_mode velocity --data_scaling linear\
       --data_path ../data/artificial/nouniform_UVC/ --image_size 200 --pc_size 100\
       --aug_rotate 0 --aug_resize 0\
       --valid_data_path ../data/artificial/nouniform_UVC/ \
       --train_path ../data/artificial_uniform_train.csv \
       --valid_path ../data/artificial_uniform_valid.csv \
       --test --eval_threshold 0.5 10 20 --test_path ../data/artificial_uniform_valid.csv \
       --result_path $case --tdim_use 12 --learning_rate 0.0002 --lr_decay 0.7 \
       --batch_size 10 --n_epochs 10 --n_threads 8 --checkpoint 10 \
       --loss_function MSE \
       --interp_type nearest \
       --optimizer adam
# png output
# python ../src_post/plot_comp_prediction_trajgru.py $case
# python ../src_post/gif_animation.py $case

