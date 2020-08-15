# codes for SageMaker Run

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

from scaler import *
from train_valid_epoch import *
from utils import Logger
from opts import parse_opts
from loss_funcs import *

# trajGRU model
from collections import OrderedDict
from models_trajGRU.network_params_trajGRU import model_structure_convLSTM, model_structure_trajGRU
from models_trajGRU.forecaster import Forecaster
from models_trajGRU.encoder import Encoder
from models_trajGRU.trajGRU import TrajGRU
from models_trajGRU.convLSTM import ConvLSTM
from models_trajGRU.loss import Weighted_mse_mae
device = torch.device("cuda")

# for SageMaker
import sagemaker
import boto3
from sagemaker.pytorch import PyTorch
    
def count_parameters(model,f):
    for name,p in model.named_parameters():
        f.write("name,"+name+", Trainable, "+str(p.requires_grad)+",#params, "+str(p.numel())+"\n")
    Nparam = sum(p.numel() for p in model.parameters())
    Ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
    f.write("Number of params:"+str(Nparam)+", Trainable parameters:"+str(Ntrain)+"\n")
    
if __name__ == '__main__':

    # set up a SageMaker session
    key_path = "../../CAUTION_aws_account_info/sagemaker_key.txt"
    with open(key_path) as f:
        lines = f.readlines()
        key_id = lines[1].replace('\n','')
        access_key = lines[2].replace('\n','')
        region = lines[3].replace('\n','')
        role = lines[4].replace('\n','')
    
    sagemaker_session = sagemaker.Session(boto3.session.Session(
        aws_access_key_id=key_id,
        aws_secret_access_key=access_key,
        region_name=region))
    bucket = sagemaker_session.default_bucket()
    prefix = 'sagemaker/trajGRU'

    local_run = False
    #local_run = True
    if local_run:
        input_data = 'file://../data/sagemaker-dir'
        instance = 'local_gpu'
    else:
        # SageMaker
        #input_data = 's3://radar-data-1906/local_transformation_model/data/sagemaker-dir'
        input_data = 's3://radar-data-1906/local_transformation_model/data/sagemaker-dir-light'
        instance = 'ml.p2.xlarge'

#    env = {
#        'SAGEMAKER_REQUIREMENTS': 'requirements.txt', 
#    }
    
    hparam_dict = {
        'model_name':'trajgru_el',
        'dataset':'radarJMA',
        'model_mode':'run',
        'data_scaling':'linear',
        'data_path':'data_kanto/',
        'image_size':200,
        'pc_size':100,
        'valid_data_path':'data_kanto/ ',
        'train_path':'train_simple_JMARadar.csv ',
        'valid_path':'valid_simple_JMARadar.csv ',
        'test_path':'valid_simple_JMARadar.csv ',
        'result_path':' result_tstrun',
        'tdim_use': 12,
        'tdim_loss': 1,
        'learning_rate': 0.0001,
        'lr_decay':0.8 ,
        'batch_size':10,
        'n_epochs':1,
        'checkpoint':10 ,
        'loss_function':'MSE',
        'interp_type':'nearest',
        }

    #Fitting

    estimator = PyTorch(entry_point="main_trajGRU_jma_sagemaker.py",
                        source_dir="../src",
                        role=role,
#                        framework_version='1.4.0',
                        framework_version='1.5.0',
                        py_version='py3',
                        instance_count=1,
                        instance_type=instance,
                        volume_size=60,
                        hyperparameters=hparam_dict)

    estimator.fit({'training': input_data})
    import pdb;pdb.set_trace()

