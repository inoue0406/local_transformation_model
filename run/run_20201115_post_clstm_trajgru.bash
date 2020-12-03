#!/bin/bash

python ../src_post/plot_pred_radarJMA_trajGRU.py result_20201004_trajGRU_size200_alljapan_mse_lr0002 trajgru
python ../src_post/gif_animation.py result_20201004_trajGRU_size200_alljapan_mse_lr0002

python ../src_post/plot_pred_radarJMA_trajGRU.py result_20201004_traj_clstm_size200_alljapan_mse_lr0002 clstm
python ../src_post/gif_animation.py result_20201004_traj_clstm_size200_alljapan_mse_lr0002

python ../src_post/plot_pred_radarJMA_trajGRU.py result_20201025_trajGRU_size200_alljapan_wmse_lr0002_w1111 trajgru
python ../src_post/gif_animation.py result_20201025_trajGRU_size200_alljapan_wmse_lr0002_w1111

##python ../src_post/plot_pred_radarJMA_trajGRU.py result_20201025_trajGRU_size200_alljapan_wmse_lr0002_w1112 trajgru
##python ../src_post/gif_animation.py result_20201025_trajGRU_size200_alljapan_wmse_lr0002_w1112
##
##python ../src_post/plot_pred_radarJMA_trajGRU.py result_20201025_trajGRU_size200_alljapan_wmse_lr0002_w1121 trajgru
##python ../src_post/gif_animation.py result_20201025_trajGRU_size200_alljapan_wmse_lr0002_w1121
##
##python ../src_post/plot_pred_radarJMA_trajGRU.py result_20201025_trajGRU_size200_alljapan_wmse_lr0002_w1211 trajgru
##python ../src_post/gif_animation.py result_20201025_trajGRU_size200_alljapan_wmse_lr0002_w1211
##
##python ../src_post/plot_pred_radarJMA_trajGRU.py result_20201025_trajGRU_size200_alljapan_wmse_lr0002_w1221 trajgru
##python ../src_post/gif_animation.py result_20201025_trajGRU_size200_alljapan_wmse_lr0002_w1221
##
##python ../src_post/plot_pred_radarJMA_trajGRU.py result_20201025_trajGRU_size200_alljapan_wmse_lr0002_w2111 trajgru
##python ../src_post/gif_animation.py result_20201025_trajGRU_size200_alljapan_wmse_lr0002_w2111
#
#python ../src_post/plot_pred_radarJMA_trajGRU.py result_20201025_trajGRU_size200_alljapan_wmse_lr0002_w2112 trajgru
#python ../src_post/gif_animation.py result_20201025_trajGRU_size200_alljapan_wmse_lr0002_w2112
#
#python ../src_post/plot_pred_radarJMA_trajGRU.py result_20201025_trajGRU_size200_alljapan_wmse_lr0002_w2121 trajgru
#python ../src_post/gif_animation.py result_20201025_trajGRU_size200_alljapan_wmse_lr0002_w2121
#
#python ../src_post/plot_pred_radarJMA_trajGRU.py result_20201025_trajGRU_size200_alljapan_wmse_lr0002_w2211 trajgru
#python ../src_post/gif_animation.py result_20201025_trajGRU_size200_alljapan_wmse_lr0002_w2211
