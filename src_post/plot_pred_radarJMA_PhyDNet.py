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

from jma_pytorch_dataset import *
from scaler import *
from train_valid_epoch import *
from colormap_JMA import Colormap_JMA

# PhyDNet model
from models_PhyDNet.models import ConvLSTM, PhyCell, EncoderRNN
device = torch.device("cuda")

def mod_str_interval(inte_str):
    # a tweak for decent filename 
    inte_str = inte_str.replace('(','')
    inte_str = inte_str.replace(']','')
    inte_str = inte_str.replace(',','')
    inte_str = inte_str.replace(' ','_')
    return(inte_str)

# plot comparison of predicted vs ground truth
def plot_comp_prediction(data_path,filelist,model_name,model_fname,batch_size,tdim_use,
                         df_sampled,pic_path,scl,case,img_size,mode='png_whole'):
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
    if model_name == 'phydnet':
        print ("PhyDNet model selected")
        phycell =  PhyCell(input_shape=(25,25), input_dim=64, F_hidden_dims=[49], n_layers=1, kernel_size=(7,7), device=device) 
        convlstm =  ConvLSTM(input_shape=(25,25), input_dim=64, hidden_dims=[128,128,64], n_layers=3, kernel_size=(3,3), device=device)
        encoder = EncoderRNN(phycell, convlstm, device)
    # load weights
    encoder.load_state_dict(torch.load(model_fname))
    # evaluation mode
    encoder.eval()

    # tmp force 2nd order to zero
    #encoder.phycell.cell_list[0].F.conv1.weight[2,:,:,:]=0.0
    #encoder.phycell.cell_list[0].F.conv1.weight[8,:,:,:]=0.0
    #encoder.phycell.cell_list[0].F.conv1.weight[14,:,:,:]=0.
    
    # tmp force even order to zero
    #for i in range(1,49):
    #    if i % 2 == 0:
    #        print("remove i=",i)
    #        encoder.phycell.cell_list[0].F.conv1.weight[i,:,:,:]=0.0
    
    # tmp everything to zero
    #for i in range(0,49):
    #    encoder.phycell.cell_list[0].F.conv1.weight[i,:,:,:]=0.0
        
    for i_batch, sample_batched in enumerate(valid_loader):
        fnames = sample_batched['fnames_future']
        # skip if no overlap
        if len(set(fnames) &  set(df_sampled['fname'].values)) == 0:
            print("skipped batch:",i_batch)
            continue
        # apply the trained model to the data
        input_tensor = Variable(scl.fwd(sample_batched['past'])).cuda()
        target_tensor = Variable(scl.fwd(sample_batched['future'])).cuda()

        # prediction
        input_length = tdim_use
        target_length = tdim_use

        for ei in range(input_length-1):
            encoder_output, encoder_hidden, _,_,_  = encoder(input_tensor[:,ei,:,:,:], (ei==0))

        decoder_input = input_tensor[:,-1,:,:,:] # first decoder input= last image of input sequence
        predictions = []

        for di in range(target_length):
            decoder_output, decoder_hidden, output_image,_,_ = encoder(decoder_input, False, False)
            decoder_input = output_image
            predictions.append(output_image)

        output =  torch.stack(predictions,dim=1) # for MM: [batch_size, time, channels, H, W]

        # Output only selected data in df_sampled
        for n,fname in enumerate(fnames):
            if (not (fname in df_sampled['fname'].values)):
                print('skipped:',fname)
                continue
            # convert to cpu
            pic = target_tensor[n,:,0,:,:].cpu()
            pic_tg = scl.inv(pic.data.numpy())
            pic = output[n,:,0,:,:].cpu()
            pic_pred = scl.inv(pic.data.numpy())
            # print
            print('Plotting: ',fname,np.max(pic_tg),np.max(pic_pred))
            # plot
            cm = Colormap_JMA()
            if mode == 'png_whole': # output as stationary image
                fig, ax = plt.subplots(figsize=(20, 10))
                fig.suptitle("Precip prediction starting at: "+fname+"\n"+case, fontsize=20)
                for nt in range(6):
                #for nt in range(1,12,2):
                    id = nt*2+1
                    pos = nt+1
                    dtstr = str((id+1)*5)
                    # target
                    plt.subplot(2,6,pos)
                    #im = plt.imshow(pic_tg[id,:,:],vmin=0,vmax=50,cmap=cm,origin='lower')
                    im = plt.imshow(pic_tg[id,:,:].transpose(),vmin=0,vmax=50,cmap=cm,origin='lower')
                    plt.title("true:"+dtstr+"min")
                    plt.grid()
                    # predicted
                    plt.subplot(2,6,pos+6)
                    #im = plt.imshow(pic_pred[id,:,:],vmin=0,vmax=50,cmap=cm,origin='lower')
                    im = plt.imshow(pic_pred[id,:,:].transpose(),vmin=0,vmax=50,cmap=cm,origin='lower')
                    plt.title("pred:"+dtstr+"min")
                    plt.grid()
                fig.subplots_adjust(right=0.95)
                cbar_ax = fig.add_axes([0.96, 0.15, 0.01, 0.7])
                fig.colorbar(im, cax=cbar_ax)
                # save as png
                i = df_sampled.index[df_sampled['fname']==fname]
                i = int(i.values)
                interval = df_sampled['rcategory'].iloc[i]
                str_int = 'I{}-{}_'.format(abs(interval.left),interval.right)
                plt.savefig(pic_path+'comp_pred_'+str_int+fname+'.png')
                plt.close()
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
                    im = plt.imshow(pic_tg[id,:,:],vmin=0,vmax=50,cmap=cm,origin='lower')
                    plt.title("true:"+dtstr+"min")
                    plt.grid()
                    # predicted
                    plt.subplot(1,2,2)
                    im = plt.imshow(pic_pred[id,:,:],vmin=0,vmax=50,cmap=cm,origin='lower')
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
        del input_tensor,target_tensor,output
        
if __name__ == '__main__':
    # params
    batch_size = 4
    tdim_use = 12
    img_size = 200

    # read case name from command line
    argvs = sys.argv
    argc = len(argvs)

    if argc != 3:
        print('Usage: python plot_comp_prediction.py CASENAME clstm/trajgru')
        quit()

    case = argvs[1]
    #case = 'result_20190625_clstm_lrdecay07_ep20'

    model_name = argvs[2]
    #model_name = 'phydnet'

    #data_path = '../data/data_kanto_resize/'
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
    
    plot_comp_prediction(data_path,filelist,model_name,model_fname,batch_size,tdim_use,
                         df_sampled,pic_path,scl,case,img_size,mode='png_ind')


