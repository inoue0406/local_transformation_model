
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py

import glob
import gzip
import subprocess
import sys
import os.path

import datetime

# pre-selected 
slct_id = np.array([[2, 1],
                    [2, 2],
                    [3, 2],
                    [3, 3],
                    [4, 2],
                    [4, 3],
                    [4, 4],
                    [4, 5],
                    [4, 6],
                    [4, 7],
                    [5, 6],
                    [5, 7],
                    [5, 8],
                    [5, 9],
                    [6, 6],
                    [6, 7],
                    [6, 8],
                    [6, 9],
                    [7, 7],
                    [7, 8],
                    [7, 9],
                    [7, 10],
                    [8, 8],
                    [8, 9],
                    [8, 10],
                    [8, 11],
                    [8, 12],
                    [8, 13],
                    [9, 9 ],
                    [9, 10],
                    [9, 11],
                    [9, 12],
                    [9, 13],
                    [10, 12],
                    [10, 13]])

def load_avg_field(infile_root,ii,jj):
    print("load averaged fields")

    ijstr = "_%d_%d.h5" % (ii,jj)
    rain_X_list = []
    files_list = sorted(glob.iglob(infile_root + "*" + ijstr))
    for infile in files_list:
        print("reading:",infile)
        h5file = h5py.File(infile,'r')
        rain_X = h5file['R'][()].astype(np.float16)
        rain_X = np.maximum(rain_X,0) # replace negative value with 0
        rain_X_list.append(rain_X)
    # stack in 0 dimension
    rain_X_all = np.stack(rain_X_list,axis=0)
    return rain_X_all,files_list

def multi_scale_avg(Vavg,flist,fname):
    flist = [fname.split("/")[-1] for fname in flist]
    # averaging at multiple time-scale
    f2 = fname.split("/")[-1]
    ijstr = "utc_%d_%d.h5" % (ii,jj)
    f2 = f2.replace("2p-jmaradar5_","").replace(ijstr,"")
    basedate = datetime.datetime.strptime(f2,'%Y-%m-%d_%H%M')
    
    Nexp = 10 # number of exponents 2^Nexp
    startdate = datetime.datetime(2015, 1, 1)
    mindate = basedate - datetime.timedelta(hours=2**(Nexp-1))
    if mindate < startdate:
        print("skipped: there is no record for averaging",basedate)
        return None
    Vmsa_list = []
    for Nexp in range(Nexp):
        time_avg = range(1,2**Nexp)
        print("time averaging with hours:",time_avg)
        date_list = [basedate - datetime.timedelta(hours=x) for x in time_avg]
        favg_list = [date.strftime('1havg_2p-jmaradar5_%Y-%m-%d_%H%M'+ijstr) for date in date_list]
        #
        #df_fname = pd.DataFrame(,columns=["fname"])
        intersect = list(set(favg_list) & set(flist))
        id_list = []
        for x in intersect:
            id_list.append(flist.index(x))
        Vaa = Vavg[id_list,:,:]
        Vaa = np.mean(Vaa,axis=0)
        Vmsa_list.append(Vaa)
        
    # whole-time averaging
    Vmsa_list.append(np.mean(Vavg,axis=0))
    # stack in 0 dimension
    Vmsa = np.stack(Vmsa_list,axis=0)
    return Vmsa

def prep_multi_average(infile_root,outfile_root,anno_list,ii,jj):
    print("preparing for the region= ",ii,jj)

    # read averaged field
    Vavg,flist = load_avg_field(infile_root,ii,jj)

    # loop through data
    ijstr = "_%d_%d.h5" % (ii,jj)
    anno_list_ij = anno_list.loc[anno_list["fname"].str.contains(ijstr)]
    
    for index, row in anno_list_ij.iterrows():
        print
        Vmsa = multi_scale_avg(Vavg,flist,row["fname"])
        if Vmsa is None:
            continue
        h5fname = row["fname"]
        print('writing h5 file:',h5fname)
        h5file = h5py.File(outfile_root+h5fname,'w')
        h5file.create_dataset("Ravg",data= Vmsa)
        h5file.close()
    
if __name__ == '__main__':
    year = 2015

    # read case name from command line
    argvs = sys.argv
    argc = len(argvs)

    if argc != 3:
        print('Usage: python plot_comp_prediction.py istart iend')
        quit()
    istart = int(argvs[1])
    iend = int(argvs[2])
    
    # read
    infile_root = "../data/data_alljapan_averaged/"
    print('input dir:',infile_root)

    # outfile
    outfile_root = '../data/data_alljapan_multiavg/'
    print('output dir:',outfile_root)

    # list of data
    anno_path = "../data/train_alljapan_2yrs_JMARadar.csv"
    anno_list = pd.read_csv(anno_path)

    # prepare grid by grid
    #for n in range(slct_id.shape[0]):
    for n in range(istart,iend):
        ii,jj = slct_id[n,:]
        prep_multi_average(infile_root,outfile_root,anno_list,ii,jj)
