from torch import nn
import torch.nn.functional as F
import torch
from models_trajGRU.utils import make_layers

import numpy as np

import rev_bilinear
from src_rev_bilinear.rev_bilinear_interp import RevBilinear
from nearest_neighbor_interp import nearest_neighbor_interp

class activation():

    def __init__(self, act_type, negative_slope=0.2, inplace=True):
        super().__init__()
        self._act_type = act_type
        self.negative_slope = negative_slope
        self.inplace = inplace

    def __call__(self, input):
        if self._act_type == 'leaky':
            return F.leaky_relu(input, negative_slope=self.negative_slope, inplace=self.inplace)
        elif self._act_type == 'relu':
            return F.relu(input, inplace=self.inplace)
        elif self._act_type == 'sigmoid':
            return torch.sigmoid(input)
        else:
            raise NotImplementedError

def xy_grid(height,width):
    # generate constant xy grid
    x1grd = torch.linspace(0,1,width).cuda() # 1d grid
    y1grd = torch.linspace(0,1,height).cuda() # 1d grid

    Xgrid = torch.zeros(height, width)
    Ygrid = torch.zeros(height, width)
    for j in range(height):
        Xgrid[j,:] = x1grd
    for k in range(width):
        Ygrid[:,k] = y1grd
    
    return Xgrid,Ygrid

def grid_to_pc(R_grd,XY_pc):
    # convert grid to pc
    # R_grd: grid value with [batch,channels,height,width] dim
    # XY_pc: point cloud position with [batch,2,N] dim
    #        scaled to [0,1]
    B, _, N = XY_pc.size()
    _, C, _, _ = R_grd.size()
    input = R_grd
    L = int(np.sqrt(N)) # note that sqrt(N) should be an integer
    vgrid = XY_pc.permute(0, 2, 1).reshape(B, L, L, 2).cuda()
    # rescale grid to [-1,1]
    vgrid = (vgrid - 0.5) * 2.0
    R_pc = torch.nn.functional.grid_sample(input, vgrid)
    R_pc = R_pc.reshape(B, C, N)
    return R_pc

def grid_to_pc_nearest(R_grd,XY_pc,XY_grd):
    # convert grid to pc
    # R_grd: grid value with [batch,channels,height,width] dim
    # XY_pc: point cloud position with [batch,2,N] dim
    #        scaled to [0,1]
    batch,k,height,width = R_grd.shape
    XY_grd_tmp = XY_grd.reshape(batch,2,height*width).permute(0,2,1).detach()
    XY_pc_tmp = XY_pc.permute(0,2,1).detach()
    R_grd_tmp = R_grd.reshape(batch,k,height*width)
    R_pc = nearest_neighbor_interp(XY_pc_tmp,XY_grd_tmp,R_grd_tmp)
    #import pdb;pdb.set_trace()
    return R_pc

def pc_to_grid(R_pc,XY_pc,height):
    # convert pc to grid
    # R_pc: point cloud value with [batch,channels,N] dim
    # XY_pc: point cloud position with [batch,2,N] dim
    #        scaled to [0,1]
    
    # apply interpolation
    R_grd = RevBilinear.apply(XY_pc, R_pc, height)
    R_grd = R_grd.permute(0,1,3,2)
    return R_grd

def pc_to_grid_nearest(R_pc,XY_pc,XY_grd):
    # convert pc to grid
    # R_pc: point cloud value with [batch,channels,N] dim
    # XY_pc: point cloud position with [batch,2,N] dim
    #        scaled to [0,1]
    batch,_,height,width = XY_grd.shape
    _,k,_ = R_pc.shape
    XY_grd_tmp = XY_grd.reshape(batch,2,height*width).permute(0,2,1).detach()
    XY_pc_tmp = XY_pc.permute(0,2,1).detach()
    R_grd = nearest_neighbor_interp(XY_pc_tmp,XY_grd_tmp,R_pc)
    R_grd = R_grd.reshape(batch,k,height,width)
    return R_grd

class EF_el(nn.Module):
    # Changed to Use Euler-Lagrange Forecaster

    def __init__(self, encoder, forecaster, image_size, batch_size,
                 mode="run", interp_type="bilinear"):
        super().__init__()
        self.encoder = encoder
        self.forecaster = forecaster
        # grid
        self.Xgrid,self.Ygrid = xy_grid(image_size,image_size)
        # mode
        self.mode = mode
        self.interp_type = interp_type

    def forward(self, input):
        
        # Grid variable: XY_grd, R_grd
        # Point cloud variable: XY_pc, R_pc
        
        bsize, tsize, channels, height, width = input.size()
        # permute to match radarJMA dataset
        input_p = input.permute((1, 0, 2, 3, 4))
        # 
        state = self.encoder(input_p)
        output = self.forecaster(state)
        # calc UV as an output of trajGRU
        output = output.permute((1, 0, 2, 3, 4))
        # use first 2 dims as uv
        UV_grd = output[:,:,[0,1],:,:]
        UV_grd =  (UV_grd - 0.5)*5.0
        # use last 1 dims as coeff
        C_grd = output[:,:,[2],:,:]
        #return output

        if self.mode=="velocity":
            # directly return velocity
            print('max_uv',torch.max(UV_grd).cpu().detach().numpy(),'min_uv',torch.min(UV_grd).cpu().detach().numpy())
            return UV_grd

        # rescale UV to [0-1] grid
        UV_grd[:,:,1,:,:] = UV_grd[:,:,1,:,:]/height
        UV_grd[:,:,0,:,:] = UV_grd[:,:,0,:,:]/width

        # Lagrangian prediction
        R_grd = input[:,-1,:,:,:] #use initial

        # Set Initial Grid (which will be fixed through time progress)
        X_grd = torch.stack(bsize*[self.Xgrid]).unsqueeze(1)
        Y_grd = torch.stack(bsize*[self.Ygrid]).unsqueeze(1)
        XY_grd = torch.cat([X_grd,Y_grd],dim=1).cuda()
        # Set Initial PC
        XY_pc = XY_grd.clone().reshape(bsize,2,height*width)
        R_pc = R_grd.reshape(bsize,1,height*width)

        xout = torch.zeros(bsize, tsize, channels, height, width,  requires_grad=True).cuda()
        if self.mode == "check":
            r_pc_out = torch.zeros(bsize, tsize, channels, height*width).cuda()
            xy_pc_out = torch.zeros(bsize, tsize, 2, height*width).cuda()
        xzero = torch.zeros(bsize, channels, height, width,  requires_grad=True).cuda() # ! should I put zero here?
        
        for it in range(tsize):
            # UV has [batch, 2, height width] dimension
            # Interpolate UV to Point Cloud position
            if self.interp_type == "bilinear":
                UV_pc = grid_to_pc(UV_grd[:,it,:,:,:],XY_pc)
                C_pc = grid_to_pc(C_grd[:,it,:,:,:],XY_pc)
            elif self.interp_type == "nearest":
                UV_pc = grid_to_pc_nearest(UV_grd[:,it,:,:,:],XY_pc,XY_grd)
                C_pc = grid_to_pc_nearest(C_grd[:,it,:,:,:],XY_pc,XY_grd)
                
            print('max_uv',torch.max(UV_pc).cpu().detach().numpy(),'min_uv',torch.min(UV_pc).cpu().detach().numpy())
            # Calc Time Progress
            XY_pc = XY_pc + UV_pc
            XY_pc = torch.clamp(XY_pc,min=0.0,max=1.0) # XY should be in [0,1]
            R_pc = R_pc * C_pc # multiplicative
            R_pc = torch.clamp(R_pc,min=0.0,max=1.0) # R_pc should be in [0,1]
            # Interpolate PC to Grid
            if self.interp_type == "bilinear":
                R_grd = pc_to_grid(R_pc,XY_pc,height)
            elif self.interp_type == "nearest":
                R_grd = pc_to_grid_nearest(R_pc,XY_pc,XY_grd)

            xout[:,it,:,:,:] = R_grd
            if self.mode == "check":
                r_pc_out[:,it,:,:] = R_pc
                xy_pc_out[:,it,:,:] = XY_pc

        if self.mode == "run":
            return xout
        elif self.mode == "check":
            # rescale back UV
            UV_grd[:,:,1,:,:] = UV_grd[:,:,1,:,:]*height
            UV_grd[:,:,0,:,:] = UV_grd[:,:,0,:,:]*width
            return xout,UV_grd,r_pc_out,xy_pc_out

class Predictor(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.model = make_layers(params)

    def forward(self, input):
        '''
        input: S*B*1*H*W
        :param input:
        :return:
        '''
        input = input.squeeze(2).permute((1, 0, 2, 3))
        output = self.model(input)
        return output.unsqueeze(2).permute((1, 0, 2, 3, 4))
