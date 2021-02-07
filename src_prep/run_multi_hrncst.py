#---------------------------------------------------
# Preprocess high-resolution nowcast data by JMA
#---------------------------------------------------
import glob
import subprocess
import sys
import os.path

import pandas as pd

if __name__ == '__main__':

    date1 = "2017-07-01"
    date2 = "2017-09-30"

    dates = pd.date_range(start=date1,end=date2)

    for i in range(len(dates)):
        datestr =  str(dates[i])[0:10]
        print("processing date:",datestr)
        for h in range(24):
            # run prep routine
            hourstr = "_%02d" % h
            command = "python ../src_prep/prep_hrncst_JMA_alljapan.py "+datestr+hourstr
            #command = "python ../src_prep/prep_hrncst_JMA_alljapan.py "+datestr
            print(command)
            subprocess.run(command,shell=True)

