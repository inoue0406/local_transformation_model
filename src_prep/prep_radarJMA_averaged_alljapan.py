# Reading JMA radar data in netcdf format
# take statistics for the whole Japanese region
import netCDF4
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

# extract data from nc file
def ext_nc_JMA(fname):
    try:
        nc = netCDF4.Dataset(fname, 'r')
    except:
        print("netCDF read error")
        return None
    eps = 0.001 # small number
    #
    # dimensions
    nx = len(nc.dimensions['LON'])
    ny = len(nc.dimensions['LAT'])
    nt = len(nc.dimensions['TIME'])
    print("dims:",nx,ny,nt)
    #
    # ext variable
    lons = nc.variables['LON'][:]
    lats = nc.variables['LAT'][:]
    R = nc.variables['PRATE'][:]
    #
    R = R.T   # transpose so that i index comes first
    return(R[:,:,0])

def fname_1h_ago(fname):
    f2 = fname.split("/")[-1]
    f2 = f2.replace("2p-jmaradar5_","").replace("utc.nc.gz","")
    dt = datetime.datetime.strptime(f2,'%Y-%m-%d_%H%M')
    
    # +1h data for Y
    date1 = dt - pd.offsets.Hour()
    #fname1 = date1.strftime('../data/jma_radar/%Y/%m/%d/2p-jmaradar5_%Y-%m-%d_%H%Mutc.nc.gz')
    fname1 = date1.strftime('/data/nas_data/jma_radar/%Y/2p-jmaradar5_%Y-%m-%d_%H%Mutc.nc.gz')
    return fname1

if __name__ == '__main__':
    #for year in [2015,2016,2017]:
    year = 2015
    #year = 2010
    #year = 2011
    #year = 2012
    #year = 2013
    #year = 2014
    #year = 2018
    #year = 2019

    # read
    infile_root = "/data/nas_data/jma_radar/%d/" % (year)
    print('input dir:',infile_root)

    # outfile
    outfile_root = '../data/data_alljapan_averaged/'
    print('output dir:',outfile_root)

    for infile in sorted(glob.iglob(infile_root + '*/*/*00utc.nc.gz')):
        # read 1hour data at a time
        # initialize with -999.0
        R_list = []
        for i in range(12):
            shour = '{0:02d}utc'.format(i*5)
            in_zfile = infile.replace('00utc',shour)
            print('reading zipped file:',in_zfile)
            # '-k' option for avoiding removing gz file
            subprocess.run('gunzip -kf '+in_zfile,shell=True)
            in_nc=in_zfile.replace('.gz','')
            print('reading nc file:',in_nc)
            if os.path.exists(in_nc):
                R_list.append(ext_nc_JMA(in_nc))
                if R_list is None:
                    next
            else:
                print('nc file not found!!!',in_nc)
                next
            subprocess.run('rm '+in_nc,shell=True)
        # create 1h data
        print("number of data in an hour",len(R_list))
        Rhour = np.stack(R_list,axis=0)
        R1h = np.mean(Rhour,axis=0)
        # create masked array for saving space
        R1hm = np.ma.masked_array(R1h,mask=R1h<-30000.0)
        # save 1h data as h5 file
        outfile = infile.split("/")[-1]
        outfile = "1havg_" + outfile.replace("nc.gz","h5")
        print('writing h5 file:',outfile)
        h5file = h5py.File(outfile_root+outfile,'w')
        #h5file.create_dataset('R',data= R1hm)
        h5file.create_dataset('R',data= R1hm,compression="gzip")
        h5file.close() 

