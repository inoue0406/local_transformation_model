#!/bin/bash

case="result_20201025_traj_ploss_faiss_size200_pc200_12step_lr0002_wMSE_1111_alljapan"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_trajGRU_jma_ploss.py --model_name trajgru_ploss\
       --dataset radarJMA --model_mode run --data_scaling linear\
       --data_path ../data/data_alljapan/ --image_size 200 --pc_size 200\
       --valid_data_path ../data/data_kanto/ \
       --train_path ../data/train_alljapan_2yrs_JMARadar.csv \
       --valid_path ../data/valid_simple_JMARadar.csv\
       --test --eval_threshold 0.5 10 20 --test_path ../data/valid_simple_JMARadar.csv \
       --result_path $case --tdim_use 12 --tdim_loss 12 --learning_rate 0.001 --lr_decay 0.9 \
       --batch_size 10 --n_epochs 20 --n_threads 4 --checkpoint 5 \
       --loss_function WeightedMSE --loss_weights 2 5 10 30\
       --interp_type nearest_faiss\
       --hidden_channels 5 --kernel_size 3 --optimizer adam 
#       --transfer_path  ../run/result_20200915_artfield_traj_el_faiss_size200_pc100_12step_UVC/trained_CLSTM.dict
# post plot
python ../src_post/plot_pred_radarJMA_traj-el.py $case nearest_faiss
python ../src_post/gif_animation.py $case

#case="result_20201012_traj_ploss_faiss_size200_pc200_12step_lr0002_WMSE_UVCneg_alljapan"
#
## running script for Rainfall Prediction with ConvLSTM
#python ../src/main_trajGRU_jma_ploss.py --model_name trajgru_ploss\
#       --dataset radarJMA --model_mode run --data_scaling linear\
#       --data_path ../data/data_alljapan/ --image_size 200 --pc_size 200\
#       --valid_data_path ../data/data_kanto/ \
#       --train_path ../data/train_alljapan_2yrs_JMARadar.csv \
#       --valid_path ../data/valid_simple_JMARadar.csv\
#       --test --eval_threshold 0.5 10 20 --test_path ../data/valid_simple_JMARadar.csv \
#       --result_path $case --tdim_use 12 --tdim_loss 12 --learning_rate 0.0001 --lr_decay 0.9 \
#       --batch_size 10 --n_epochs 20 --n_threads 4 --checkpoint 10 \
#       --loss_function WeightedMSE\
#       --interp_type nearest_faiss\
#       --hidden_channels 5 --kernel_size 3 --optimizer adam \
#       --transfer_path  ../run/result_20200915_artfield_traj_el_faiss_size200_pc100_12step_UVC/trained_CLSTM.dict
## post plot
#python ../src_post/plot_pred_radarJMA_traj-el.py $case nearest_faiss
#python ../src_post/gif_animation.py $case


