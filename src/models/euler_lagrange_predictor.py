# 
# Hybrid Eulerian Lagrangian Rainfall Prediction
# 

import torch
import torch.nn as nn
#from torch.autograd import Variable

import numpy as np

# -----------------------------
# add "src" as import path
#path = os.path.join('/home/tsuyoshi/local_transformation_model/src/src_rev_bilinear/')
#sys.path.append(path)

#import models.convolution_lstm_mod
import rev_bilinear
from src_rev_bilinear.rev_bilinear_interp import RevBilinear

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = torch.zeros(1, hidden, shape[0], shape[1],requires_grad=True).cuda()
            self.Wcf = torch.zeros(1, hidden, shape[0], shape[1],requires_grad=True).cuda()
            self.Wco = torch.zeros(1, hidden, shape[0], shape[1],requires_grad=True).cuda()
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (torch.zeros(batch_size, hidden, shape[0], shape[1],requires_grad=True).cuda(),
                torch.zeros(batch_size, hidden, shape[0], shape[1],requires_grad=True).cuda())

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

class Euler_Lagrange_Predictor(nn.Module):
    # Encoder-Predictor using Eulerian Lagrangian approach
    def __init__(self, input_channels, hidden_channels, kernel_size, image_size, batch_size):
        # input_channels (scalar) 
        # hidden_channels (scalar) 
        super(Euler_Lagrange_Predictor, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self._all_layers = []
        # initialize encoder/predictor cell
        cell_e = ConvLSTMCell(self.input_channels, self.hidden_channels, self.kernel_size)
        self.encoder = cell_e
        self._all_layers.append(cell_e)
        cell_p = ConvLSTMCell(self.input_channels, self.hidden_channels, self.kernel_size)
        self.predictor = cell_p
        self._all_layers.append(cell_p)
        # last conv layer for prediction
        self.padding = int((self.kernel_size - 1) / 2)
        # convolution for obtaining (U,V)
        self.uvconv = nn.Conv2d(self.hidden_channels, 2, self.kernel_size, 1, self.padding, bias=True)
        # grid
        self.Xgrid,self.Ygrid = xy_grid(batch_size,image_size,image_size)
        
    def forward(self, input):
        x = input
        bsize, tsize, channels, height, width = x.size()
        # initialize internal state
        (he, ce) = self.encoder.init_hidden(batch_size=bsize, hidden=self.hidden_channels, shape=(height, width))
        (hp, cp) = self.predictor.init_hidden(batch_size=bsize, hidden=self.hidden_channels, shape=(height, width))
        # encoding
        for it in range(tsize):
            # forward
            (he, ce) = self.encoder(x[:,it,:,:,:], he, ce)
        # copy internal state to predictor
        hp = he
        cp = ce
        # Lagrangian prediction
        Rgrd = x[:,-1,:,:,:] #use initial

        # Set Initial PC
        X_pc = self.Xgrid.reshape(bsize,1,height*width)
        Y_pc = self.Ygrid.reshape(bsize,1,height*width)
        XY_pc = torch.cat([X_pc,Y_pc],dim=1).cuda()
        R_pc = Rgrd.reshape(bsize,1,height*width)

        xout = torch.zeros(bsize, tsize, channels, height, width,  requires_grad=True).cuda()
        #xout_prev = Rgrd
        
        for it in range(tsize):
            (hp, cp) = self.predictor(Rgrd, hp, cp) # input previous timestep's xout
            # calc velocity field from hidden layer
            UV_grd = self.uvconv(hp)
            # UV has [batch, 2, height width] dimension
            #xout_prev = xout[:,it,:,:,:].clone()
            
            # Interpolate UV to Point Cloud position
            UV_pc = grid_to_pc(UV_grd,XY_pc)
            # Calc Time Progress
            XY_pc = XY_pc + UV_pc
            XY_pc = torch.clamp(XY_pc,min=0.0,max=1.0) # XY should be in [0,1]
            # Interpolate PC to Grid
            Rgrd = pc_to_grid(R_pc,XY_pc,height)

            xout[:,it,:,:,:] = Rgrd

        return xout

