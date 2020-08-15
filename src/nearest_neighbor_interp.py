import torch

import numpy as np
from scipy.spatial import cKDTree

from multiprocessing import Pool
from multiprocessing import Process

import faiss

def batch_pairwise_distances(x, y=None):
    '''
    Input: x is a bxNxd matrix where b is batch size
           y is an optional bxMxd matirx
    Output: dist is a bxNxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    b,N,d = x.shape
    x_norm = (x**2).sum(2).view(b,-1, 1)
    if y is not None:
        y_norm = (y**2).sum(2).view(b, 1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
        
    dist = x_norm + y_norm - 2.0 * torch.bmm(x, torch.transpose(y, 1, 2))
    return dist

def nearest_neighbor_interp(xy_grd_b,xy_pc_b,R_pc_b):
    '''
    Input: xy_grd_b bxNx2 matrix where b is batch size, N is regular grid mesh size
           xy_pc_b  bxMx2 matrix where b is batch size, M is point cloud size
           R_pc_b   bxkxM matrix where b is batch size, M is point cloud size
    Output: R_grd_b  bxkxN interpolated value at grid point
    '''
    b,k,M = R_pc_b.shape
    _,N,_ = xy_grd_b.shape
    #D = batch_pairwise_distances(xy_grd_b, xy_pc_b)
    #id_min = torch.min(D,2).indices
    id_min = torch.zeros([b,N],dtype=torch.int64).cuda()
    #splits = 1000
    splits = 100
    #for n in range(N):
    #    D = batch_pairwise_distances(xy_grd_b[:,n:n+1,:], xy_pc_b)
    #    id_min[:,n:n+1] = torch.min(D,2).indices
    for nsp in np.array_split(range(N),splits):
        D = batch_pairwise_distances(xy_grd_b[:,nsp,:], xy_pc_b)
        id_min[:,nsp] = torch.min(D,2).indices
        #if(n % 1000 == 0):
        #    print("n=",n)
    # add dimension 
    #id_min = id_min[:,None,:]
    id_min = torch.stack(k*[id_min],axis=1)
    # interpolate from point cloud to grid
    R_grd_b = torch.gather(R_pc_b,2,id_min)
    return R_grd_b

# ------------------------------------------
# kdtree for faster implementation

def min_dist_KDTree(inputs):
    points_A,points_B = inputs
    # minimum distance by KDTree
    
    # create KDTree
    kd_tree = cKDTree(points_B)

    dists, indices = kd_tree.query(points_A, k=1)
    
    return indices

def multi_mindist(xy_grd_b,xy_pc_b,bs):
    # multicore 
    #inputs = [(xy_grd_b[n,:,:],xy_pc_b[n,:,:]) for n in range(bs)]
    #with Pool(5) as p:
    #   result = p.map(min_dist_KDTree,inputs)
    # single core
    result = []
    for n in range(bs):
          res = min_dist_KDTree([xy_grd_b[n,:,:],xy_pc_b[n,:,:]])
          result.append(res)
          
    return np.stack(result)

def nearest_neighbor_interp_kd(xy_grd_b,xy_pc_b,R_pc_b):
    '''
    Input: xy_grd_b bxNx2 matrix where b is batch size, N is regular grid mesh size
           xy_pc_b  bxMx2 matrix where b is batch size, M is point cloud size
           R_pc_b   bxkxM matrix where b is batch size, M is point cloud size
    Output: R_grd_b  bxkxN interpolated value at grid point
    '''
    b,k,M = R_pc_b.shape
    _,N,_ = xy_grd_b.shape
    # minimum distance by KDTree
    id_min_np = multi_mindist(xy_grd_b.cpu().detach().numpy(),
                           xy_pc_b.cpu().detach().numpy(),b)
    id_min = torch.from_numpy(id_min_np).cuda()
    # add dimension 
    #id_min = id_min[:,None,:]
    id_min = torch.stack(k*[id_min],axis=1)
    # interpolate from point cloud to grid
    R_grd_b = torch.gather(R_pc_b,2,id_min)
    return R_grd_b

# ------------------------------------------
# feiss implementation
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
    #faiss.bfKnn(res, metric,
                        xb_ptr, xb_row_major, nb,
                        xq_ptr, xq_row_major, nq,
                        d, k, D_ptr, I_ptr)

    return D, I

def nearest_neighbor_interp_fe(xy_A_b,xy_B_b,R_B_b):
    '''
    Input: xy_A_b bxNx2 matrix where b is batch size, N is regular grid mesh size
           xy_B_b  bxMx2 matrix where b is batch size, M is point cloud size
           R_B_b   bxkxM matrix where b is batch size, M is point cloud size
    Output: R_A_b  bxkxN interpolated value at grid point
    '''
    b,k,M = R_B_b.shape
    _,N,_ = xy_A_b.shape
    # minimum distance by feiss

    #method = "FlatIP"
    method = "IVF"
    
    id_min = torch.zeros([b,N],dtype=torch.int64).cuda()
    
    for n in range(b):

        points_A = xy_A_b[n,:,:].detach().contiguous()
        points_B = xy_B_b[n,:,:].detach().cpu().numpy()
        
        if method == "FlatIP":
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatIP(res, 2)
            index.add(points_B)
        elif method == "IVF":
            res = faiss.StandardGpuResources()
            nlist = 100
            quantizer = faiss.IndexFlatL2(2)  # the other index
            index_ivf = faiss.IndexIVFFlat(quantizer, 2, nlist, faiss.METRIC_L2)
            # here we specify METRIC_L2, by default it performs inner-product search
        
            # make it an IVF GPU index
            index = faiss.index_cpu_to_gpu(res, 0, index_ivf)
    
            assert not index.is_trained
            index.train(points_B)        # add vectors to the index
            assert index.is_trained
            index.add(points_B)          # add vectors to the index        

        # (1) search_index_pytorch
        # query is pytorch tensor (GPU)
        # no need for a sync here
        D, ID = search_index_pytorch(index, points_A, 1)
        res.syncDefaultStreamCurrentDevice()
        #D,ID = search_raw_array_pytorch(res,xy_B_b[n,:,:],xy_A_b[n,:,:],1)
        id_min[n,:] = ID[:,0]
        
    # add dimension 
    id_min = torch.stack(k*[id_min],axis=1)
    # interpolate from point cloud to grid
    R_A_b = torch.gather(R_B_b,2,id_min)
    return R_A_b

def set_index_faiss(points_B):
    # set FAISS index
    # points_B : numpy array with Mx2 dimension
    
    #points_B = points_B.contiguous()
    
    #method = "FlatIP"
    method = "IVF"
    if method == "FlatIP":
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatIP(res, 2)
        index.add(points_B)
    elif method == "IVF":
        res = faiss.StandardGpuResources()
        nlist = 100
        quantizer = faiss.IndexFlatL2(2)  # the other index
        index_ivf = faiss.IndexIVFFlat(quantizer, 2, nlist, faiss.METRIC_L2)
        # here we specify METRIC_L2, by default it performs inner-product search
        # make it an IVF GPU index
        index = faiss.index_cpu_to_gpu(res, 0, index_ivf)

        assert not index.is_trained
        index.train(points_B)        # add vectors to the index
        assert index.is_trained
        index.add(points_B)          # add vectors to the index
    return index

def nearest_neighbor_interp_fi(xy_A_b,xy_B_b,R_B_b,index,mode):
    '''
    Interpolation by faiss with "precalculated" index
    Input: xy_A_b bxNx2 matrix where b is batch size, N is regular grid mesh size
           xy_B_b  bxMx2 matrix where b is batch size, M is point cloud size
           R_B_b   bxkxM matrix where b is batch size, M is point cloud size
           index  faiss index
           mode   forward or backward
    Output: R_A_b  bxkxN interpolated value at grid point
    '''
    b,k,M = R_B_b.shape
    _,N,_ = xy_A_b.shape
    # minimum distance by faiss
    #print("R_B_b shape",R_B_b.shape)

    id_min = torch.zeros([b,N],dtype=torch.int64).cuda()
    
    for n in range(b):

        if mode=="forward":
            # forward : use normal query 
            points_A = xy_A_b[n,:,:].detach().contiguous()
            D, ID = search_index_pytorch(index, points_A, 1)
            #res.syncDefaultStreamCurrentDevice()
            #print("ID shape fwd",ID.shape)
            
        elif mode=="backward":
            # backward : use reverse query with forward index
            points_B = xy_B_b[n,:,:].detach().contiguous()
            k_rev = 5
            DR, IDR = search_index_pytorch(index, points_B, k_rev)
            # create distance table
            D_tbl = torch.full((N,k_rev),999.9).cuda()
            ID_tbl = torch.zeros(N,k_rev,dtype=torch.int64).cuda()
            # fill distance table
            for kk in range(k_rev):
                D_tbl[IDR[:,kk],kk] = DR[:,kk]
                ID_tbl[IDR[:,kk],kk] = torch.arange(M).cuda()
            min_index = torch.argmin(D_tbl,dim=1,keepdim=True)
            ID = ID_tbl.gather(1,min_index)
            #import pdb;pdb.set_trace()
            #print("ID shape back",ID.shape)
            
        id_min[n,:] = ID[:,0]
        
    # add dimension 
    id_min = torch.stack(k*[id_min],axis=1)
    # interpolate from point cloud to grid
    #print("id_min shape",id_min.shape)
    R_A_b = torch.gather(R_B_b,2,id_min)
    return R_A_b

if __name__ == '__main__':
    # test for distance function
    # define size
    height = 50
    width = 50
    # define position of circle
    ic = 25
    jc = 25
    # spatial scale
    scale = 5

    # create initial field
    R = torch.zeros(1,1,height,width)
    # create xy grid
    xx = torch.arange(0, width).view(1, -1).repeat(height, 1).float()
    yy = torch.arange(0, height).view(-1, 1).repeat(1, width).float()
    # create circular field
    R[0,0,:,:] = torch.exp(-((xx-ic)**2 + (yy-jc)**2)/scale**2)
    
    # Define point cloud with 2d initialization
    # point cloud size
    Npc = 1600
    # torch.rand generates uniform [0,1)
    XY = torch.rand(1, 2, Npc, dtype=torch.float32, requires_grad=True)
    XY = XY * height # Scale to uniform [0,height]
    print("xy shape",XY.shape)

    R = torch.zeros(1,1,Npc)
    # create circular field
    R[0,0,:] = torch.exp(-((XY[0,0]-ic)**2 + (XY[0,1]-jc)**2)/scale**2)

    # scale to [-1,1]
    #XYZ_scl = XYZ/25.0*2.0-1.0
    XY_scl = XY/50.0
    print(torch.min(XY_scl),torch.max(XY_scl))

    # Regular Grid: xx,yy [50,50]
    # Irregular Grid: XY [1,2,1600]
    xy_grd = torch.stack([xx,yy]).reshape(2,50*50).permute(1,0)
    print("shape of xy_grd:",xy_grd.shape)
    xy_pc = XY[0,:,:].permute(1,0)
    print("shape of xy_pc:",xy_pc.shape)
    # make example of batch size 2
    xy_grd_b = torch.stack([xy_grd,xy_grd])
    xy_pc_b = torch.stack([xy_pc,xy_pc])
    R_pc_b = torch.cat([R,R])
    RR_pc_b = torch.cat(3*[R_pc_b],dim=1)
    print("shape of xy_p_bc:",xy_pc_b.shape)

    #R_grd_b = nearest_neighbor_interp(xy_grd_b,xy_pc_b,R_pc_b)

    R_grd_b = nearest_neighbor_interp(xy_grd_b,xy_pc_b,RR_pc_b)
    
    R_grd_b2 = nearest_neighbor_interp_kd(xy_grd_b,xy_pc_b,RR_pc_b)
    
    import pdb;pdb.set_trace()
    
