#
# Local Transformation Model for spatio-temporal prediction
# 

# Modified to follow encoder-predictor structure

import torch
import torch.nn as nn
from torch.autograd import Variable

def one_step_ltm(Xin,T):
    # single step of ltm time evolution
    # Xout = Xin
    #  shift for calculating difference term
    #   0  1  2
    #   3  4  5
    #   6  7  8
    Xpad = torch.nn.functional.pad(Xin, (1, 1, 1, 1))
    X0 = Xpad[:, 0:-2, 0:-2]
    X1 = Xpad[:, 1:-1, 0:-2]
    X2 = Xpad[:, 2:, 0:-2]
    X3 = Xpad[:, 0:-2, 1:-1]
    X4 = Xpad[:, 1:-1, 1:-1]
    X5 = Xpad[:, 2:, 1:-1]
    X6 = Xpad[:, 0:-2, 2:]
    X7 = Xpad[:, 1:-1, 2:]
    X8 = Xpad[:, 2:, 2:]
    # time evolution
    Xout = X0*T[:,0,:,:] + X1*T[:,1,:,:] + X2*T[:,2,:,:] + \
           X3*T[:,3,:,:] + X4*T[:,4,:,:] + X5*T[:,5,:,:] + \
           X6*T[:,6,:,:] + X7*T[:,7,:,:] + X8*T[:,8,:,:] + \
           T[:,9,:,:]
    return Xout

class LTM_predictor(nn.Module):
    # LTM Predictor
    def __init__(self, unet, in_channels, steps_in_itr):
        super(LTM_predictor, self).__init__()
        # initialize Unet
        self.unet = unet
        self.in_channels = in_channels
        self.steps_in_itr= steps_in_itr
        
    def forward(self, input):
        x = input
        bsize, tsize, channels, height, width = x.size()
        # take the latest n series as an input
        x_in = x[:,-self.in_channels:,0,:,:]
        # x_in is [bs, in_ch, width, height] dim

        T = self.unet(x_in)
        # T is [bs, out_ch, width, height] dim

        # Time step using transformation matrix
        xout = Variable(torch.zeros(bsize, tsize, channels, height, width)).cuda()
        xt = Variable(torch.zeros(bsize, channels, height, width)).cuda()
        # start from the latest frame
        xt = one_step_ltm(x[:,-1,0,:,:],T)
        xout[:,0,0,:,:] = xt
        for i in range(1,tsize):
            # calc time step
            for j in range(0,self.steps_in_itr):
                #print("time:",i," step:",j)
                xt = one_step_ltm(xt,T)
                xt = torch.sigmoid(one_step_ltm(xt,T))
            xout[:,i,0,:,:] = xt
        return xout
    
