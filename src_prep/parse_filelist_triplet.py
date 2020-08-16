#!/usr/bin/env python

# parse data directory to make train/test list

import glob
import pandas as pd
import numpy as np
import os

import pdb
import h5py
#pdb.set_trace()

def read_h5_max(fpath):
    # read file and check the value
    h5file = h5py.File(fpath,'r')
    rain = h5file['R'].value
    h5file.close()
    return(rain.mean(),rain.max())

dirpath = "../data/data_kanto_aug"
file_list = sorted(glob.glob(dirpath))

#setting = "train"
setting = "valid"

#mode = "all"
mode = "nozero"

# initialize
mean_list = []
fnames = []
fnext1 = []
fnext2 = []

if setting == "train":
    # select 2015 and 2016 for training
    dates = pd.date_range(start='2015-01-01 00:00', end='2016-12-31 22:00', freq='H')
elif setting == "valid":
    # select 2017 for validation
    dates = pd.date_range(start='2017-01-01 00:00', end='2017-12-31 22:00', freq='H')
    
# file format "2p-jmaradar5_2015-01-01_0000utc.h5"

# We choose loop through continuous times for missed-file checking and 
# checking for valid X-Y pairs

if setting == "train":
    fcsv = '../data/filelist_triplet_train_JMARadar_%s.csv' % mode
elif setting == "valid":
    fcsv = '../data/filelist_triplet_valid_JMARadar_%s.csv' % mode
    
f = open(fcsv, mode='w')
f.write("fname, fname1, fname2, rain_mean_mm, rain_max_mm\n")

for n,date in enumerate(dates):
    print(date)
    fname = date.strftime('2p-jmaradar5_%Y-%m-%d_%H%Mutc.h5')
    print(fname)
    fpath = os.path.join(dirpath,fname)
    
    # +1h data for Y
    date1 = date + pd.offsets.Hour()
    fname1 = date1.strftime('2p-jmaradar5_%Y-%m-%d_%H%Mutc.h5')
    fpath1 = os.path.join(dirpath,fname1)

    # +2h data for Y
    date2 = date1 + pd.offsets.Hour()
    fname2 = date2.strftime('2p-jmaradar5_%Y-%m-%d_%H%Mutc.h5')
    fpath2 = os.path.join(dirpath,fname2)

    # current data
    if os.path.exists(fpath):
        print('Exists:',fpath)
        rmean,rmax = read_h5_max(fpath)
    else:
        continue
    # check for next hour
    if os.path.exists(fpath1):
        print('Exists Next Hour:',fpath1)
        rmean1,rmax1 = read_h5_max(fpath1)
    else:
        print('NOT Exist Next Hour:',fpath1)
        continue
    # check for next next hour
    if os.path.exists(fpath2):
        print('Exists Next Hour:',fpath2)
        rmean2,rmax2 = read_h5_max(fpath1)
    else:
        print('NOT Exist Next Hour:',fpath2)
        continue

    rmean_all = (rmean + rmean1 + rmean2)/3.0
    rmax_all = np.max([rmax,rmax1,rmax2])
    if mode == 'all':
        f.write("%s, %s, %s, %7.4f, %7.4f\n" % (fname,fname1,fname2,rmean_all,rmax_all))
    if mode == 'nozero':
        # output only if rain is large enough
        if rmean_all >= 0.1:
            f.write("%s, %s, %s, %7.4f, %7.4f\n" % (fname,fname1,fname2,rmean_all,rmax_all))
