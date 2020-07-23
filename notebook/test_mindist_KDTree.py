# test the functionarity of KDTree

import numpy as np
from scipy.spatial import cKDTree
import time

from multiprocessing import Pool
from multiprocessing import Process

def min_dist_KDTree(inputs):
    N,M = inputs
    # minimum distance by KDTree

    points_A = np.random.random((N, 2))
    points_B = np.random.random((M, 2))

    # create KDTree
    kd_tree = cKDTree(points_B)

    dists, indices = kd_tree.query(points_A, k=1)
    
    return indices

def multi_mindist(bs):
    p = Pool(8)
    inputs = [(N, M) for n in range(bs)]
    #import pdb;pdb.set_trace()
    result = p.map(min_dist_KDTree,inputs)
    return result

if __name__ == '__main__':
    N = 20000
    M = 20000
    bs = 10

    # no parallel version
    
    t1 = time.time()    
    for i in range(bs):
        indices = min_dist_KDTree([N,M])
        print("length of indices:",len(indices))
        
    t2 = time.time()
    dt = t2-t1
    print("elapsed time: %f" % dt)
    
    # no parallel version
    
    t1 = time.time()
    indices = multi_mindist(bs)
    #import pdb;pdb.set_trace()
    t2 = time.time()
    dt = t2-t1
    print("elapsed time: %f" % dt)
