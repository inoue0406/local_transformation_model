#!/bin/bash


case="result_20200809_traj_el_faiss_size200_pc150_1step_lr0001_ivf"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_trajGRU_jma.py --model_name trajgru_el \
       --dataset radarJMA --model_mode run --data_scaling linear\
       --data_path ../data/data_kanto/ --image_size 200 --pc_size 150\
       --valid_data_path ../data/data_kanto/ \
       --train_path ../data/train_simple_JMARadar.csv \
       --valid_path ../data/valid_simple_JMARadar.csv \
       --test --eval_threshold 0.5 10 20 --test_path ../data/valid_simple_JMARadar.csv \
       --result_path $case --tdim_use 12 --tdim_loss 1 --learning_rate 0.0001 --lr_decay 0.8 \
       --batch_size 10 --n_epochs 2 --n_threads 4 --checkpoint 10 \
       --loss_function MSE --interp_type nearest\
       --hidden_channels 5 --kernel_size 3 --optimizer adam
#       --transfer_path  ../run/result_20200615_artfield_value_traj_el_size200/trained_CLSTM.model \

# png output
#python ../src_post/plot_comp_prediction_trajgru.py $case
#python ../src_post/gif_animation.py $case

case="result_20200809_traj_el_faiss_size200_pc100_1step_lr0001_ivf"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_trajGRU_jma.py --model_name trajgru_el \
       --dataset radarJMA --model_mode run --data_scaling linear\
       --data_path ../data/data_kanto/ --image_size 200 --pc_size 100\
       --valid_data_path ../data/data_kanto/ \
       --train_path ../data/train_simple_JMARadar.csv \
       --valid_path ../data/valid_simple_JMARadar.csv \
       --test --eval_threshold 0.5 10 20 --test_path ../data/valid_simple_JMARadar.csv \
       --result_path $case --tdim_use 12 --tdim_loss 1 --learning_rate 0.0001 --lr_decay 0.8 \
       --batch_size 10 --n_epochs 2 --n_threads 4 --checkpoint 10 \
       --loss_function MSE --interp_type nearest\
       --hidden_channels 5 --kernel_size 3 --optimizer adam
#       --transfer_path  ../run/result_20200615_artfield_value_traj_el_size200/trained_CLSTM.model \

# png output
#python ../src_post/plot_comp_prediction_trajgru.py $case
#python ../src_post/gif_animation.py $case

case="result_20200809_traj_el_faiss_size200_pc200_1step_lr0001_ivf"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_trajGRU_jma.py --model_name trajgru_el \
       --dataset radarJMA --model_mode run --data_scaling linear\
       --data_path ../data/data_kanto/ --image_size 200 --pc_size 200\
       --valid_data_path ../data/data_kanto/ \
       --train_path ../data/train_simple_JMARadar.csv \
       --valid_path ../data/valid_simple_JMARadar.csv \
       --test --eval_threshold 0.5 10 20 --test_path ../data/valid_simple_JMARadar.csv \
       --result_path $case --tdim_use 12 --tdim_loss 1 --learning_rate 0.0001 --lr_decay 0.8 \
       --batch_size 10 --n_epochs 2 --n_threads 4 --checkpoint 10 \
       --loss_function MSE --interp_type nearest\
       --hidden_channels 5 --kernel_size 3 --optimizer adam
#       --transfer_path  ../run/result_20200615_artfield_value_traj_el_size200/trained_CLSTM.model \

# png output
#python ../src_post/plot_comp_prediction_trajgru.py $case
#python ../src_post/gif_animation.py $case
