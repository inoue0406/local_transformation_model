#
# Plot Predicted Rainfall Data
#
import torch
import torchvision
import numpy as np
import torch.utils.data as data
import gc

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

from jma_pytorch_dataset import *
from scaler import *
from train_valid_epoch import *
from colormap_JMA import Colormap_JMA

# trajGRU model
from collections import OrderedDict
from models_trajGRU.network_params_trajGRU import model_structure_convLSTM, model_structure_trajGRU
from models_trajGRU.forecaster import Forecaster
from models_trajGRU.encoder import Encoder
from models_trajGRU.model_euler_lagrange_ploss import EF_el_ploss
from models_trajGRU.trajGRU import TrajGRU
from models_trajGRU.convLSTM import ConvLSTM
device = torch.device("cuda")

def mod_str_interval(inte_str):
    # a tweak for decent filename 
    inte_str = inte_str.replace('(','')
    inte_str = inte_str.replace(']','')
    inte_str = inte_str.replace(',','')
    inte_str = inte_str.replace(' ','_')
    return(inte_str)

# plot comparison of predicted vs ground truth
def plot_comp_prediction(data_path,filelist,model_fname,batch_size,tdim_use,
                         df_sampled,pic_path,scl,case,img_size,interp_type,mode='png_whole'):
    # create pic save dir
    if not os.path.exists(pic_path):
        os.mkdir(pic_path)

    # dataset
    valid_dataset = JMARadarDataset(root_dir=data_path,
                                    csv_file=filelist,
                                    tdim_use=tdim_use,
                                    transform=None)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
    
    # load the saved model
    # model_ld = torch.load(model_fname)
    # load from state_dict
    model_name = "trajgru_el"
    encoder_params,forecaster_params = model_structure_trajGRU(img_size,batch_size,model_name)
    encoder = Encoder(encoder_params[0], encoder_params[1]).to(device)
    forecaster = Forecaster(forecaster_params[0], forecaster_params[1],tdim_use).to(device)
    model = EF_el_ploss(encoder, forecaster, img_size, 200, batch_size, 'eval', interp_type).to(device)
    model.load_state_dict(torch.load(model_fname))
    
    # evaluation mode
    model.eval()
    # activate check mode
    # model.mode = "check"
    #
    for i_batch, sample_batched in enumerate(valid_loader):
        fnames = sample_batched['fnames_future']
        # skip if no overlap
        if len(set(fnames) &  set(df_sampled['fname'].values)) == 0:
            print("skipped batch:",i_batch)
            continue
        # apply the trained model to the data
        input = Variable(scl.fwd(sample_batched['past'])).cuda()
        target = Variable(scl.fwd(sample_batched['future'])).cuda()
        #input = Variable(sample_batched['past']).cpu()
        #target = Variable(sample_batched['future']).cpu()
        #xout,uvout,r_pc_out,xy_pc_out = model(input)
        xout = model(input)
        
        # Output only selected data in df_sampled
        for n,fname in enumerate(fnames):
            if (not (fname in df_sampled['fname'].values)):
                print('skipped:',fname)
                continue
            # convert to cpu
            pic = target[n,:,0,:,:].cpu()
            pic_tg = scl.inv(pic.data.numpy())
            pic = xout[n,:,0,:,:].cpu()
            pic_pred = scl.inv(pic.data.numpy())
            # print
            print('Plotting: ',fname,np.max(pic_tg),np.max(pic_pred))
            # plot
            cm = Colormap_JMA()
            if mode == 'png_ind': # output as invividual image
                for nt in range(6):
                    fig, ax = plt.subplots(figsize=(8, 4))
                    fig.suptitle("Precip prediction starting at: "+fname+"\n"+case, fontsize=10)
                    #        
                    id = nt*2+1
                    pos = nt+1
                    dtstr = str((id+1)*5)
                    # target
                    plt.subplot(1,2,1)
                    #im = plt.imshow(pic_tg[id,:,:],vmin=0,vmax=50,cmap=cm,origin='lower')
                    im = plt.imshow(pic_tg[id,:,:].transpose(),vmin=0,vmax=50,cmap=cm,origin='lower')
                    plt.title("true:"+dtstr+"min")
                    plt.grid()
                    # predicted
                    plt.subplot(1,2,2)
                    #im = plt.imshow(pic_pred[id,:,:],vmin=0,vmax=50,cmap=cm,origin='lower')
                    im = plt.imshow(pic_pred[id,:,:].transpose(),vmin=0,vmax=50,cmap=cm,origin='lower')
                    plt.title("pred:"+dtstr+"min")
                    plt.grid()
                    # color bar
                    fig.subplots_adjust(right=0.93,top=0.85)
                    cbar_ax = fig.add_axes([0.94, 0.15, 0.01, 0.7])
                    fig.colorbar(im, cax=cbar_ax)
                    # save as png
                    i = df_sampled.index[df_sampled['fname']==fname]
                    i = int(i.values)
                    interval = mod_str_interval(df_sampled['rcategory'].iloc[i])
                    nt_str = '_dt%02d' % nt
                    plt.savefig(pic_path+'comp_pred_'+interval+fname+nt_str+'.png')
                    plt.close()
        # free GPU memory (* This seems to be necessary if going without backprop)
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    print(type(obj), obj.size())
                    del obj
            except: pass
        torch.cuda.empty_cache()
        #del input,target,xout,uvout,r_pc_out,xy_pc_out
        
if __name__ == '__main__':
    # params
    batch_size = 4
    tdim_use = 12
    #img_size = 128
    img_size = 200

    # read case name from command line
    argvs = sys.argv
    argc = len(argvs)

    if argc != 3:
        print('Usage: python plot_comp_prediction.py CASENAME INTERP_TYPE')
        quit()

    case = argvs[1]
    interp_type = argvs[2]
    #case = 'result_20190712_tr_clstm_flatsampled'
    #case = 'result_20190625_clstm_lrdecay07_ep20'

    if img_size == 128:
        data_path = '../data/data_kanto_resize/'
    elif img_size == 200:
        data_path = '../data/data_kanto/'
    filelist = '../data/valid_simple_JMARadar.csv'
    model_fname = case + '/trained_CLSTM.dict'
    pic_path = case + '/png/'

    data_scaling = 'linear'
    
    # prepare scaler for data
    if data_scaling == 'linear':
        scl = LinearScaler()
    if data_scaling == 'root':
        scl = RootScaler()
    if data_scaling == 'root_int':
        scl = RootIntScaler()
    elif data_scaling == 'log':
        scl = LogScaler()

    # samples to be plotted
    sample_path = '../data/sampled_forplot_3day_JMARadar.csv'

    # read sampled data in csv
    df_sampled = pd.read_csv(sample_path)
    print('samples to be plotted')
    print(df_sampled)
    
    plot_comp_prediction(data_path,filelist,model_fname,batch_size,tdim_use,
                         df_sampled,pic_path,scl,case,img_size,interp_type,mode='png_ind')


