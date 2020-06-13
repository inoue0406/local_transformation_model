from torch import nn
import torch.nn.functional as F
import torch
from models_trajGRU.utils import make_layers

import numpy as np

import rev_bilinear
from src_rev_bilinear.rev_bilinear_interp import RevBilinear

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

def xy_grid(bsize,height,width):
    # generate constant xy grid
    x1grd = torch.linspace(0,1,width).cuda() # 1d grid
    y1grd = torch.linspace(0,1,height).cuda() # 1d grid

    Xgrid = torch.zeros(bsize, height, width)
    Ygrid = torch.zeros(bsize, height, width)
    for i in range(bsize):
        for j in range(height):
            Xgrid[i,j,:] = x1grd
        for k in range(width):
            Ygrid[i,:,k] = y1grd
    
    return Xgrid,Ygrid

def grid_to_pc(Rgrd,XY_pc):
    # convert grid to pc
    # Rgrd: grid value with [batch,channels,height,width] dim
    # XY_pc: point cloud position with [batch,2,N] dim
    #        scaled to [0,1]
    B, C, N = XY_pc.size()
    input = Rgrd
    L = int(np.sqrt(N)) # note that sqrt(N) should be an integer
    vgrid = XY_pc.permute(0, 2, 1).reshape(B, L, L, 2).cuda()
    # rescale grid to [-1,1]
    vgrid = (vgrid - 0.5) * 2.0
    R_pc = torch.nn.functional.grid_sample(input, vgrid)
    R_pc = R_pc.reshape(B, 2, N)
    return R_pc

def pc_to_grid(R_pc,XY_pc,height):
    # convert pc to grid
    # R_pc: point cloud value with [batch,channels,N] dim
    # XY_pc: point cloud position with [batch,2,N] dim
    #        scaled to [0,1]
    
    # apply interpolation
    Rgrd = RevBilinear.apply(XY_pc, R_pc, height)
    return Rgrd

class EF_el(nn.Module):
    # Changed to Use Euler-Lagrange Forecaster

    def __init__(self, encoder, forecaster, image_size, batch_size, mode="run"):
        super().__init__()
        self.encoder = encoder
        self.forecaster = forecaster
        # grid
        self.Xgrid,self.Ygrid = xy_grid(batch_size,image_size,image_size)
        # mode
        self.mode = mode

    def forward(self, input):
        bsize, tsize, channels, height, width = input.size()
        # permute to match radarJMA dataset
        input_p = input.permute((1, 0, 2, 3, 4))
        # 
        state = self.encoder(input_p)
        output = self.forecaster(state)
        # calc UV as an output of trajGRU
        UV_grd =  output.permute((1, 0, 2, 3, 4))
        #return output

        # Lagrangian prediction
        Rgrd = input[:,-1,:,:,:] #use initial

        # Set Initial PC
        X_pc = self.Xgrid.reshape(bsize,1,height*width)
        Y_pc = self.Ygrid.reshape(bsize,1,height*width)
        XY_pc = torch.cat([X_pc,Y_pc],dim=1).cuda()
        R_pc = Rgrd.reshape(bsize,1,height*width)

        xout = torch.zeros(bsize, tsize, channels, height, width,  requires_grad=True).cuda()
        if self.mode == "check":
            uvout = torch.zeros(bsize, tsize, 2, height, width).cuda()
            r_pc_out = torch.zeros(bsize, tsize, channels, height*width).cuda()
            xy_pc_out = torch.zeros(bsize, tsize, 2, height*width).cuda()
        xzero = torch.zeros(bsize, channels, height, width,  requires_grad=True).cuda() # ! should I put zero here?
        
        for it in range(tsize):
            # UV has [batch, 2, height width] dimension
            # Interpolate UV to Point Cloud position
            UV_pc = grid_to_pc(UV_grd[:,it,:,:,:],XY_pc)
            print('max_uv',torch.max(UV_pc),'min_uv',torch.min(UV_pc))
            # Calc Time Progress
            XY_pc = XY_pc + UV_pc*10.0
            XY_pc = torch.clamp(XY_pc,min=0.0,max=1.0) # XY should be in [0,1]
            # Interpolate PC to Grid
            Rgrd = pc_to_grid(R_pc,XY_pc,height)

            xout[:,it,:,:,:] = Rgrd
            if self.mode == "check":
                uvout[:,it,:,:,:] = UV_grd
                r_pc_out[:,it,:,:] = R_pc
                xy_pc_out[:,it,:,:] = XY_pc

        if self.mode == "run":
            return xout
        elif self.mode == "check":
            return xout,uvout,r_pc_out,xy_pc_out

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
