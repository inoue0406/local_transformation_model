# test the functionarity of FAISS

import numpy as np
from scipy.spatial import cKDTree
import time

import faiss
import torch

def swig_ptr_from_FloatTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.float32
    return faiss.cast_integer_to_float_ptr(
        x.storage().data_ptr() + x.storage_offset() * 4)

def swig_ptr_from_LongTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.int64, 'dtype=%s' % x.dtype
    return faiss.cast_integer_to_long_ptr(
        x.storage().data_ptr() + x.storage_offset() * 8)

def search_index_pytorch(index, x, k, D=None, I=None):
    """call the search function of an index with pytorch tensor I/O (CPU
    and GPU supported)"""
    assert x.is_contiguous()
    n, d = x.size()
    assert d == index.d
    
    if D is None:
        D = torch.empty((n, k), dtype=torch.float32, device=x.device)
    else:
        assert D.size() == (n, k)
        
    if I is None:
        I = torch.empty((n, k), dtype=torch.int64, device=x.device)
    else:
        assert I.size() == (n, k)
    torch.cuda.synchronize()
    xptr = swig_ptr_from_FloatTensor(x)
    Iptr = swig_ptr_from_LongTensor(I)
    Dptr = swig_ptr_from_FloatTensor(D)
    index.search_c(n, xptr,
                   k, Dptr, Iptr)
    torch.cuda.synchronize()
    return D, I

def search_raw_array_pytorch(res, xb, xq, k, D=None, I=None,
                             metric=faiss.METRIC_L2):
    assert xb.device == xq.device

    nq, d = xq.size()
    if xq.is_contiguous():
        xq_row_major = True
    elif xq.t().is_contiguous():
        xq = xq.t()    # I initially wrote xq:t(), Lua is still haunting me :-)
        xq_row_major = False
    else:
        raise TypeError('matrix should be row or column-major')

    xq_ptr = swig_ptr_from_FloatTensor(xq)

    nb, d2 = xb.size()
    assert d2 == d
    if xb.is_contiguous():
        xb_row_major = True
    elif xb.t().is_contiguous():
        xb = xb.t()
        xb_row_major = False
    else:
        raise TypeError('matrix should be row or column-major')
    xb_ptr = swig_ptr_from_FloatTensor(xb)

    if D is None:
        D = torch.empty(nq, k, device=xb.device, dtype=torch.float32)
    else:
        assert D.shape == (nq, k)
        assert D.device == xb.device

    if I is None:
        I = torch.empty(nq, k, device=xb.device, dtype=torch.int64)
    else:
        assert I.shape == (nq, k)
        assert I.device == xb.device

    D_ptr = swig_ptr_from_FloatTensor(D)
    I_ptr = swig_ptr_from_LongTensor(I)

    faiss.bruteForceKnn(res, metric,
                        xb_ptr, xb_row_major, nb,
                        xq_ptr, xq_row_major, nq,
                        d, k, D_ptr, I_ptr)

    return D, I

def to_column_major(x):
    if hasattr(torch, 'contiguous_format'):
        return x.t().clone(memory_format=torch.contiguous_format).t()
    else:
        # was default setting before memory_format was introduced
        return x.t().clone().t()

if __name__ == '__main__':
    N = 20000
    M = 20000
    bs = 10
    d = 2

    points_A = np.random.randn(N, 2).astype('float32')
    points_B = np.random.randn(M, 2).astype('float32')
    #points_A = faiss.randn(N * d, 1234).reshape(N, d)
    #points_B = faiss.randn(M * d, 1235).reshape(M, d)
    t1 = time.time()

    #import pdb;pdb.set_trace()

    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatIP(res, 2)
    index.add(points_B)

    pa_torch = torch.from_numpy(points_A)
    pa_torch = pa_torch.cuda()
    pb_torch = torch.from_numpy(points_B)
    pb_torch = pb_torch.cuda()

    # (1) search_index_pytorch
    # query is pytorch tensor (GPU)
    # no need for a sync here
    #D3, I3 = search_index_pytorch(index, pa_torch, 1)
    #res.syncDefaultStreamCurrentDevice()
    
    # (2) raw_array_search
    # resource object, can be re-used over calls
    res = faiss.StandardGpuResources()
    # put on same stream as pytorch to avoid synchronizing streams
    res.setDefaultNullStreamAllDevices()
    
    D, I = search_raw_array_pytorch(res, pb_torch, pa_torch, 1)
    
    t2 = time.time()
    dt = t2-t1
    print("elapsed time: %f" % dt)
