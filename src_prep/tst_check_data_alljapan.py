
import h5py
import numpy as np

#fnames = ["data_alljapan_2014/2p-jmaradar5_2014-09-26_1100utc_4_2.h5",
#          "data_alljapan_2014/2p-jmaradar5_2014-09-18_0300utc_8_8.h5",
#          "data_alljapan_2014/2p-jmaradar5_2014-03-28_0300utc_4_3.h5",
#          "data_alljapan_2014/2p-jmaradar5_2014-03-12_1900utc_8_9.h5",
#          "data_alljapan_2014/2p-jmaradar5_2014-11-18_2100utc_4_2.h5",
#          "data_alljapan_2014/2p-jmaradar5_2014-03-20_0900utc_8_12.h5",
#          "data_alljapan_2014/2p-jmaradar5_2014-09-08_0600utc_7_9.h5",
#          "data_alljapan_2014/2p-jmaradar5_2014-07-13_0700utc_10_12.h5",
#          "data_alljapan_2014/2p-jmaradar5_2014-11-26_1800utc_5_7.h5",
#          "data_alljapan_2014/2p-jmaradar5_2014-05-28_0900utc_7_7.h5"]
#
#
#fnext = ["data_alljapan_2018/2p-jmaradar5_2018-10-23_2300utc_9_12.h5",
#          "data_alljapan_2014/2p-jmaradar5_2014-08-31_1800utc_4_3.h5",
#          "data_alljapan_2018/2p-jmaradar5_2018-09-19_1600utc_3_2.h5",
#          "data_alljapan_2016/2p-jmaradar5_2016-08-17_0700utc_3_2.h5",
#          "data_alljapan_2011/2p-jmaradar5_2011-06-23_1500utc_8_10.h5",
#          "data_alljapan_2011/2p-jmaradar5_2011-06-27_1300utc_4_7.h5",
#          "data_alljapan_2016/2p-jmaradar5_2016-08-17_0600utc_8_8.h5",
#          "data_alljapan_2012/2p-jmaradar5_2012-05-29_1700utc_9_9.h5",
#          "data_alljapan_2012/2p-jmaradar5_2012-04-25_1300utc_6_8.h5",o
#          "data_alljapan_2015/2p-jmaradar5_2015-09-27_1000utc_9_11.h5"]

fnames = ["data_alljapan_2014/2p-jmaradar5_2014-05-28_0900utc_7_7.h5"]

fnext = ["data_alljapan_2014/2p-jmaradar5_2014-05-28_1000utc_7_7.h5"]

for i in range(len(fnames)):
    
    h5_name_X = "../data/" + fnames[i]
    h5file = h5py.File(h5_name_X,'r')
    rain_X = h5file['R'][()].astype(np.float32)
    
    h5_name_Y = "../data/" + fnext[i]
    h5file = h5py.File(h5_name_Y,'r')
    rain_Y = h5file['R'][()].astype(np.float32)

    print("X: max min",np.min(rain_X),np.max(rain_X),rain_X.shape)
    print("Y: max min",np.min(rain_Y),np.max(rain_Y),rain_Y.shape)
    import pdb;pdb.set_trace()
    
