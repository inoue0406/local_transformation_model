#!/bin/bash

case="result_20201003_traj_ploss_faiss_size200_pc200_12step_lr0002_MSE_UVCneg_data3"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_trajGRU_jma_ploss.py --model_name trajgru_ploss\
       --dataset radarJMA3 --model_mode run --data_scaling linear\
       --data_path ../data/data_kanto_aug/ --image_size 200 --pc_size 200\
       --valid_data_path ../data/data_kanto_aug/ \
       --train_path ../data/filelist_triplet_train_JMARadar_nozero.csv \
       --valid_path ../data/filelist_triplet_valid_JMARadar_nozero.csv \
       --test --eval_threshold 0.5 10 20 --test_path ../data/filelist_triplet_valid_JMARadar_nozero.csv \
       --result_path $case --tdim_use 12 --tdim_loss 12 --learning_rate 0.00001 --lr_decay 0.9 \
       --batch_size 10 --n_epochs 20 --n_threads 4 --checkpoint 10 \
       --loss_function MSE\
       --interp_type nearest_faiss\
       --hidden_channels 5 --kernel_size 3 --optimizer adam \
       --transfer_path  ../run/result_20200930_artfield_nouniform_predUV_traj_el_size200/trained_CLSTM.dict
# post plot
python ../src_post/plot_pred_radarJMA_traj-el.py $case nearest_faiss
python ../src_post/gif_animation.py $case

