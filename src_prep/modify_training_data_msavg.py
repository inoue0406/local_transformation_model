# 
# 
# 
import os
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # modify training/test file list according to multi-scale data availability

    # directory of averaged dataset
    outfile_root = '../data/data_alljapan_multiavg/'
    print('output dir:',outfile_root)

    # original list of data
    anno_path = "../data/train_alljapan_2yrs_JMARadar.csv"
    anno_list = pd.read_csv(anno_path)
    
    # output
    #output_path = "../data/train_alljapan_2yrs_JMARadar_msavg.csv"
    output_path = anno_path.replace(".csv","_msavg.csv")

    # list of
    files_list = sorted(glob.iglob(outfile_root + "*.h5"))

    id_found = []

    for index, row in anno_list.iterrows():
        if any(row["fname"] in s for s in files_list):
            print("found",row["fname"])
            id_found.append(index)
        else:
            print("NOT found",row["fname"])
        if index > 1000:
            break
    
    anno_list_slct = anno_list.iloc[id_found]
    print("size of original dataset",len(anno_list))
    print("size of processed dataset",len(anno_list_slct))

    anno_list_slct.to_csv(output_path,index=False)

    import pdb;pdb.set_trace()
