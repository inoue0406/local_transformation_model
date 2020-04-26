#!/bin/bash

case="result_20200426_LTM_size128_lrdecay09"

# trial script for local transformation network
python ../src/main_LTM_jma.py --data_path ../data/data_kanto_resize/ \
       --valid_data_path ../data/data_kanto_resize/ \
       --train_path ../data/train_simple_JMARadar.csv \
       --valid_path ../data/valid_simple_JMARadar.csv \
       --test --eval_threshold 0.5 10 20 --test_path ../data/valid_simple_JMARadar.csv \
       --result_path $case --tdim_use 12 --learning_rate 0.002 --lr_decay 0.9 \
       --batch_size 60 --n_epochs 20 --n_threads 4 --checkpoint 10 \
       --optimizer adam --model_name ltm --in_channels 2

