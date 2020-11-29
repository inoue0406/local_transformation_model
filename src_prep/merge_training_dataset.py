# 
# Merge Training Datasets from various sources
# 
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Setting
    data_root = "../data/"
    data_dirs = ["DWD_RYDL_reduced",
                 "TAASRAD19",
                 "data_alljapan"]
    file_list = ["DWD_RYDL_train_200_reduced.csv",
                 "TAASRAD19_train_200_reduced.csv",
                 "train_alljapan_2yrs_JMARadar.csv"]
    fout = "JMA_DWD_TAASRAD_combined_train.csv"

    # initialize with empty dataframe
    df_all = pd.DataFrame()

    for n in range(len(data_dirs)):
        data_dir = data_dirs[n]
        train_fname = file_list[n]
        print("data_dir=",data_dir)
        print("train_fname=",train_fname)
        df = pd.read_csv(data_root+train_fname)
        # add relative path
        df_tmp = pd.DataFrame({"fanme":data_dir + "/" + df["fname"],
                              "fnext":data_dir + "/" + df["fnext"]})
        df_all = df_all.append(df_tmp)
        
    # save to csv file
    df_all = df_all.reset_index(drop=True)
    df_all.to_csv(data_root+fout)
