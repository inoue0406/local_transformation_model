#
# Plot Predicted Rainfall Data
#
import torch
import torchvision
import numpy as np
import torch.utils.data as data

import pandas as pd
import h5py
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# -----------------------------
# add "src" as import path
path = os.path.join('../src')
sys.path.append(path)

from artfield_pytorch_dataset import *
from scaler import *
from train_valid_epoch import *

# trajGRU model
from collections import OrderedDict
from models_trajGRU.forecaster import Forecaster
from models_trajGRU.encoder import Encoder
from models_trajGRU.model import EF
from models_trajGRU.trajGRU import TrajGRU
from models_trajGRU.convLSTM import ConvLSTM
device = torch.device("cuda")

# plot a field
def plot_field(X,title,png_fpath,vmin=0,vmax=1):
    plt.imshow(X,vmin=vmin,vmax=vmax,cmap="GnBu",origin='lower')
    plt.colorbar()
    plt.grid()
    plt.title(title)
    plt.show()
    plt.savefig(png_fpath)
    plt.close()
    
# plot comparison of predicted vs ground truth
def plot_comp_prediction(data_path,filelist,model_fname,batch_size,tdim_use,
                         pic_path,scl,case,mode='png_whole'):
    # create pic save dir
    if not os.path.exists(pic_path):
        os.mkdir(pic_path)

    # dataset
    valid_dataset = ArtfieldDataset(root_dir=data_path,
                                    csv_file=filelist,
                                    mode="run",
                                    tdim_use=tdim_use,
                                    transform=None)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
    # load the saved model
    model = torch.load(model_fname)
    #model = CLSTM_EP(input_channels=1, hidden_channels=12,
    #                    kernel_size=3).cuda()
    #model.load_state_dict(torch.load(model_fname))
    # evaluation mode
    model.eval()
    #
    for i_batch, sample_batched in enumerate(valid_loader):
        fnames = sample_batched['fnames']
        # apply the trained model to the data
        input = Variable(scl.fwd(sample_batched['past'])).cuda()
        target = Variable(scl.fwd(sample_batched['future'])).cuda()
        #input = Variable(sample_batched['past']).cpu()
        #target = Variable(sample_batched['future']).cpu()
        output = model(input)
    
        for n,fname in enumerate(fnames):
            # convert to cpu
            pic = target[n,:,0,:,:].cpu()
            pic_tg = scl.inv(pic.data.numpy())
            pic = output[n,:,0,:,:].cpu()
            pic_pred = scl.inv(pic.data.numpy())
            # print
            print('Plotting: ',fname,np.max(pic_tg),np.max(pic_pred))
            
            for nt in range(12):
                nt_str = '_dt%02d' % nt
                png_fpath = pic_path+'comp_pred_'+str(n)+'_tg'+nt_str+'.png'
                # plot ground truth
                plot_field(pic_tg[nt,:,:],
                                  fname+' ground truth:'+nt_str,png_fpath)
                # plot UV
                png_fpath = pic_path+'comp_pred_'+str(n)+'_pred'+nt_str+'.png'
                plot_field(pic_pred[nt,:,:],
                                  fname+' model prediction:'+nt_str,png_fpath)
        # free GPU memory (* This seems to be necessary if going without backprop)
        del input,target,output
        return
        
if __name__ == '__main__':
    # params
    batch_size = 10
    tdim_use = 12

    # read case name from command line
    argvs = sys.argv
    argc = len(argvs)

    if argc != 2:
        print('Usage: python plot_comp_prediction.py CASENAME')
        quit()

    case = argvs[1]

    data_path = '../data/artificial/uniform/'
    filelist = '../data/artificial_uniform_valid.csv'
    model_fname = case + '/trained_CLSTM.model'
    pic_path = case + '/png/'

    data_scaling = 'linear'
    
    # prepare scaler for data
    if data_scaling == 'linear':
        # use identity scaler
        scl = LinearScaler(rmax=1.0)
        
    plot_comp_prediction(data_path,filelist,model_fname,batch_size,tdim_use,
                         pic_path,scl,case,mode='png_ind')


