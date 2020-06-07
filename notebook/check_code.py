import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# -----------------------------
# add "src" as import path
path = os.path.join('/home/tsuyoshi/local_transformation_model/src/src_rev_bilinear/')
sys.path.append(path)
path = os.path.join('/home/tsuyoshi/local_transformation_model/src/')
sys.path.append(path)

#import models.convolution_lstm_mod
import rev_bilinear
from rev_bilinear_interp import RevBilinear

from models.euler_lagrange_predictor import Euler_Lagrange_Predictor

kernel_size = 3
image_size = 128
batch_size = 1
hidden_channels = 8
model = Euler_Lagrange_Predictor(input_channels=1, hidden_channels=hidden_channels,
                                 kernel_size=kernel_size,image_size=image_size,batch_size=batch_size,mode="check").cuda()
#model = Euler_Lagrange_Predictor(input_channels=1, hidden_channels=hidden_channels,
#                                 kernel_size=kernel_size,image_size=image_size,batch_size=batch_size).cuda()

# define size
height = image_size
width = image_size
# define position of circle
ic = 50
jc = 50
# spatial scale
scale = 25
# time steps
tsize = 12

# create initial field
R = torch.zeros(1,tsize,1,height,width).cuda()
# create xy grid
xx = torch.arange(0, width).view(1, -1).repeat(height, 1).float()
yy = torch.arange(0, height).view(-1, 1).repeat(1, width).float()
# create circular field
for n in range(tsize):
        R[0,n,0,:,:] = torch.exp(-((xx-ic)**2 + (yy-jc)**2)/scale**2)

#output = model(R)
xout,uvout,r_pc_out,xy_pc_out = model(R)

import pdb;pdb.set_trace()
