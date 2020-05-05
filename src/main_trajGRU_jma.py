import numpy as np
import torch 
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms

import pandas as pd
import h5py
import os
import sys
import json
import time

import pdb

from jma_pytorch_dataset import *
from scaler import *
from train_valid_epoch import *
from utils import Logger
from opts import parse_opts
from loss_funcs import *

# trajGRU model
from collections import OrderedDict
from models_trajGRU.forecaster import Forecaster
from models_trajGRU.encoder import Encoder
from models_trajGRU.model import EF
from models_trajGRU.trajGRU import TrajGRU
from models_trajGRU.convLSTM import ConvLSTM
from models_trajGRU.model import activation
from models_trajGRU.loss import Weighted_mse_mae
device = torch.device("cuda")
ACT_TYPE = activation('leaky', negative_slope=0.2, inplace=True)

def count_parameters(model,f):
    for name,p in model.named_parameters():
        f.write("name,"+name+", Trainable, "+str(p.requires_grad)+",#params, "+str(p.numel())+"\n")
    Nparam = sum(p.numel() for p in model.parameters())
    Ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
    f.write("Number of params:"+str(Nparam)+", Trainable parameters:"+str(Ntrain)+"\n")
    
if __name__ == '__main__':
   
    # parse command-line options
    opt = parse_opts()
    print(opt)
    # create result dir
    if not os.path.exists(opt.result_path):
        os.mkdir(opt.result_path)
    
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    # generic log file
    logfile = open(os.path.join(opt.result_path, 'log_run.txt'),'w')
    logfile.write('Start time:'+time.ctime()+'\n')
    tstart = time.time()

    # model information
    modelinfo = open(os.path.join(opt.result_path, 'model_info.txt'),'w')


    # model structure
    # build model for convlstm
    # modified for 128x128
    convlstm_encoder_params = [
        [
            OrderedDict({'conv1_leaky_1': [1, 8, 7, 3, 1]}),
            OrderedDict({'conv2_leaky_1': [64, 192, 5, 3, 1]}),
            OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
        ],

        [
            ConvLSTM(input_channel=8, num_filter=64, b_h_w=(opt.batch_size, 42, 42),
                     kernel_size=3, stride=1, padding=1),
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(opt.batch_size, 14, 14),
                     kernel_size=3, stride=1, padding=1),
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(opt.batch_size, 7, 7),
                     kernel_size=3, stride=1, padding=1),
        ]
    ]

    convlstm_forecaster_params = [
        [
            OrderedDict({'deconv1_leaky_1': [192, 192, 4, 2, 1]}),
            OrderedDict({'deconv2_leaky_1': [192, 64, 5, 3, 1]}),
            OrderedDict({
                'deconv3_leaky_1': [64, 8, 7, 3, 1],
                'conv3_leaky_2': [8, 8, 3, 1, 1],
                'conv3_3': [8, 1, 1, 1, 0]
            }),
        ],

        [
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(opt.batch_size, 7, 7),
                     kernel_size=3, stride=1, padding=1),
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(opt.batch_size, 14, 14),
                     kernel_size=3, stride=1, padding=1),
            ConvLSTM(input_channel=64, num_filter=64, b_h_w=(opt.batch_size, 42, 42),
                     kernel_size=3, stride=1, padding=1),
        ]
    ]
    # parameters for trajGRU
    # build model
    # 
    trajgru_encoder_params = [
        [
            OrderedDict({'conv1_leaky_1': [1, 8, 7, 3, 1]}),
            OrderedDict({'conv2_leaky_1': [64, 192, 5, 3, 1]}),
            OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
        ],
        
        [
            TrajGRU(input_channel=8, num_filter=64, b_h_w=(opt.batch_size, 42, 42), zoneout=0.0, L=13,
                    i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                    h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                    act_type=ACT_TYPE),
            
            TrajGRU(input_channel=192, num_filter=192, b_h_w=(opt.batch_size, 14, 14), zoneout=0.0, L=13,
                    i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                    h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                    act_type=ACT_TYPE),
            TrajGRU(input_channel=192, num_filter=192, b_h_w=(opt.batch_size, 7, 7), zoneout=0.0, L=9,
                    i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                    h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                    act_type=ACT_TYPE)
        ]
    ]
    trajgru_forecaster_params = [
        [
            OrderedDict({'deconv1_leaky_1': [192, 192, 4, 2, 1]}),
            OrderedDict({'deconv2_leaky_1': [192, 64, 5, 3, 1]}),
            OrderedDict({
                'deconv3_leaky_1': [64, 8, 7, 3, 1],
                'conv3_leaky_2': [8, 8, 3, 1, 1],
                'conv3_3': [8, 1, 1, 1, 0]
            }),
        ],

        [
            TrajGRU(input_channel=192, num_filter=192, b_h_w=(opt.batch_size, 7, 7), zoneout=0.0, L=13,
                    i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                    h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                    act_type=ACT_TYPE),

            TrajGRU(input_channel=192, num_filter=192, b_h_w=(opt.batch_size, 14, 14), zoneout=0.0, L=13,
                    i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                    h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                    act_type=ACT_TYPE),
            TrajGRU(input_channel=64, num_filter=64, b_h_w=(opt.batch_size, 42, 42), zoneout=0.0, L=9,
                    i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                    h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                    act_type=ACT_TYPE)
        ]
    ]

    # prepare scaler for data
    if opt.data_scaling == 'linear':
        scl = LinearScaler()
    if opt.data_scaling == 'root':
        scl = RootScaler()
    if opt.data_scaling == 'root_int':
        scl = RootIntScaler()
    elif opt.data_scaling == 'log':
        scl = LogScaler()
        
    if not opt.no_train:
        # loading datasets
        train_dataset = JMARadarDataset(root_dir=opt.data_path,
                                        csv_file=opt.train_path,
                                        tdim_use=opt.tdim_use,
                                        transform=None)
    
        valid_dataset = JMARadarDataset(root_dir=opt.valid_data_path,
                                        csv_file=opt.valid_path,
                                        tdim_use=opt.tdim_use,
                                        transform=None)
    
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=opt.batch_size,
                                                   num_workers=4,
                                                   drop_last=True,
                                                   shuffle=True)
    
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                   batch_size=opt.batch_size,
                                                   num_workers=4,
                                                   drop_last=True,
                                                   shuffle=False)

        if opt.model_name == 'clstm':
            # convolutional lstm
            encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1]).to(device)
            forecaster = Forecaster(convlstm_forecaster_params[0], convlstm_forecaster_params[1]).to(device)
            model = EF(encoder, forecaster).to(device)
        elif opt.model_name == 'trajgru':
            # trajGRU model
            encoder = Encoder(trajgru_encoder_params[0], trajgru_encoder_params[1]).to(device)
            forecaster = Forecaster(trajgru_forecaster_params[0], trajgru_forecaster_params[1]).to(device)
            model = EF(encoder, forecaster).to(device)
    
        if opt.transfer_path != 'None':
            # Use pretrained weights for transfer learning
            print('loading pretrained model:',opt.transfer_path)
            model = torch.load(opt.transfer_path)

        modelinfo.write('Model Structure \n')
        modelinfo.write(str(model))
        count_parameters(model,modelinfo)
        modelinfo.close()
        
        if opt.loss_function == 'MSE':
            loss_fn = torch.nn.MSELoss()
        elif opt.loss_function == 'Weighted_MSE_MAE':
            loss_fn = Weighted_mse_mae(LAMBDA=0.01).to(device)

        # Type of optimizers adam/rmsprop
        if opt.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
        elif opt.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=opt.learning_rate)
            
        # learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=opt.lr_decay)
            
        # Prep logger
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'loss', 'lr'])
        valid_logger = Logger(
            os.path.join(opt.result_path, 'valid.log'),
            ['epoch', 'loss'])
    
        # training 
        for epoch in range(1,opt.n_epochs+1):
            # step scheduler
            scheduler.step()
            # training & validation
            train_epoch(epoch,opt.n_epochs,train_loader,model,loss_fn,optimizer,
                        train_logger,train_batch_logger,opt,scl)
            #valid_epoch(epoch,opt.n_epochs,valid_loader,model,loss_fn,
            #            valid_logger,opt,scl)

            if epoch % opt.checkpoint == 0:
                # save the trained model for every checkpoint
                # (1) as binary 
                torch.save(model,os.path.join(opt.result_path,
                                                 'trained_CLSTM_epoch%03d.model' % epoch))
                # (2) as state dictionary
                torch.save(model.state_dict(),
                           os.path.join(opt.result_path,
                                        'trained_CLSTM_epoch%03d.dict' % epoch))
        # save the trained model
        # (1) as binary 
        torch.save(model,os.path.join(opt.result_path, 'trained_CLSTM.model'))
        # (2) as state dictionary
        torch.save(model.state_dict(),
                   os.path.join(opt.result_path, 'trained_CLSTM.dict'))

    # test datasets if specified
    if opt.test:
        if opt.no_train:
            #load pretrained model from results directory
            model_fname = os.path.join(opt.result_path, 'trained_CLSTM.model')
            print('loading pretrained model:',model_fname)
            model = torch.load(model_fname)
            loss_fn = torch.nn.MSELoss()
            
        # prepare loader
        test_dataset = JMARadarDataset(root_dir=opt.valid_data_path,
                                        csv_file=opt.test_path,
                                        tdim_use=opt.tdim_use,
                                        transform=None)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                   batch_size=opt.batch_size,
                                                   num_workers=4,
                                                   drop_last=True,
                                                   shuffle=False)

        
        
        # testing for the trained model
        for threshold in opt.eval_threshold:
            test_CLSTM_EP(test_loader,model,loss_fn,opt,scl,threshold)

    # output elapsed time
    logfile.write('End time: '+time.ctime()+'\n')
    tend = time.time()
    tdiff = float(tend-tstart)/3600.0
    logfile.write('Elapsed time[hours]: %f \n' % tdiff)
