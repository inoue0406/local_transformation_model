# 
# Hybrid Eulerian Lagrangian Rainfall Prediction
# 

import torch
import torch.nn as nn
from torch.autograd import Variable

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
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda())

def xy_grid(bsize,height,width):
    # generate constant xy grid
    x1grd = torch.linspace(0,1,width).cuda() # 1d grid
    y1grd = torch.linspace(0,1,height).cuda() # 1d grid

    Xgrid = torch.zeros(bsize, 1, height, width)
    Ygrid = torch.zeros(bsize, 1, height, width)
    for i in range(bsize):
        for j in range(height):
            Xgrid[i,0,j,:] = x1grd
        for k in range(width):
            Ygrid[i,0,:,k] = y1grd
    
    return Xgrid,Ygrid

# input: B, C, H, W
# flow: [B, 2, H, W]
device = torch.device("cuda")
def warp(input, flow):
    B, C, H, W = input.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1).to(device)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W).to(device)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    vgrid = grid + flow

    # scale grid to [-1,1] for input to grid_sample
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(input, vgrid)
    return output

class Euler_Lagrange_Const_Predictor(nn.Module):
    # Encoder-Predictor using Eulerian Lagrangian approach
    def __init__(self, input_channels, hidden_channels, kernel_size):
        # input_channels (scalar) 
        # hidden_channels (scalar) 
        super(Euler_Lagrange_Const_Predictor, self).__init__()
        self.input_channels = input_channels + 2
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
        self.Xgrid,self.Ygrid = xy_grid(10,128,128)
        
    def forward(self, input):
        x = input
        bsize, tsize, channels, height, width = x.size()
        # initialize internal state
        (he, ce) = self.encoder.init_hidden(batch_size=bsize, hidden=self.hidden_channels, shape=(height, width))
        (hp, cp) = self.predictor.init_hidden(batch_size=bsize, hidden=self.hidden_channels, shape=(height, width))
        # initialize gridded vector field
        UV = Variable(torch.zeros(bsize, 2, height, width)).cuda()
        # encoding
        for it in range(tsize):
            # forward
            # coord convolution
            xtemp = torch.cat((x[:,it,:,:,:],self.Xgrid.cuda() ,self.Ygrid.cuda()),dim=1)
            (he, ce) = self.encoder(xtemp, he, ce)
        # calc velocity field from hidden layer
        UV = self.uvconv(he)
        # UV has [batch, 2, height width] dimension
        # Lagrangian prediction
        Rgrd = x[:,-1,:,:,:] #use initial

        xout = Variable(torch.zeros(bsize, tsize, channels, height, width)).cuda()
        for it in range(tsize):
            warped = warp(Rgrd, (it+1)*UV)
            xout[:,it,:,:,:] = warped

        return xout

