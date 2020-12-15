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
    data_dirs = ["data_alljapan_2010",
                 "data_alljapan_2011",
                 "data_alljapan_2012",
                 "data_alljapan_2013",
                 "data_alljapan_2015",
                 "data_alljapan_2016",
                 "data_alljapan_2018",
                 "data_alljapan_2019"]
    file_list = ["train_alljapan_2010_JMARadar.csv",
                 "train_alljapan_2011_JMARadar.csv",
                 "train_alljapan_2012_JMARadar.csv",
                 "train_alljapan_2013_JMARadar.csv",
                 "train_alljapan_2015_JMARadar.csv",
                 "train_alljapan_2016_JMARadar.csv",
                 "train_alljapan_2018_JMARadar.csv",
                 "train_alljapan_2019_JMARadar.csv"]
    fout = "JMA_8years_combined_train.csv"

    # initialize with empty dataframe
    df_all = pd.DataFrame()

    for n in range(len(data_dirs)):
        data_dir = data_dirs[n]
        train_fname = file_list[n]
        print("data_dir=",data_dir)
        print("train_fname=",train_fname)
        df = pd.read_csv(data_root+train_fname)
        # add relative path
        df_tmp = pd.DataFrame({"fname":data_dir + "/" + df["fname"],
                              "fnext":data_dir + "/" + df["fnext"]})
        df_all = df_all.append(df_tmp)
        
    # save to csv file
    df_all = df_all.reset_index(drop=True)
    df_all.to_csv(data_root+fout)
