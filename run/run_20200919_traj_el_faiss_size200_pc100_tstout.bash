#!/bin/bash

#case="result_20200919_traj_el_faiss_size200_pc100_tstout_normal"

# post plot
#python ../src_post/plot_pred_radarJMA_traj-el.py $case nearest_normal

#
#case="result_20200919_traj_el_faiss_size200_pc100_tstout_kdtree"
#
## post plot
#python ../src_post/plot_pred_radarJMA_traj-el.py $case nearest_kdtree
#

case="result_20200919_traj_el_faiss_size200_pc100_tstout_faiss"

# post plot
python ../src_post/plot_pred_radarJMA_traj-el.py $case nearest_faiss


