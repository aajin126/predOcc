#!/usr/bin/env python
#
# file: $ISIP_EXP/SOGMP/scripts/model.py
#
# revision history: xzt
#  20220824 (TE): first version
#
# usage:
#
# This script hold the model architecture
#------------------------------------------------------------------------------

# import pytorch modules
#
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from convlstm import ConvLSTMCell
import matplotlib.pyplot as plt
# import modules
#
import os
import random

# for reproducibility, we seed the rng
#
SEED1 = 1337
NEW_LINE = "\n"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-----------------------------------------------------------------------------
#
# helper functions are listed here
#
#-----------------------------------------------------------------------------

# function: set_seed
#
# arguments: seed - the seed for all the rng
#
# returns: none
#
# this method seeds all the random number generators and makes
# the results deterministic
#
def set_seed(seed):
    #torch.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #random.seed(seed)
    #os.environ['PYTHONHASHSEED'] = str(seed)
#
# end of method


# function: get_data
#
# arguments: img_path - file pointer
#            file_name - the name of data file
#
# returns: data - the signals/features
#
# this method takes in a fp and returns the data and labels
POINTS = 1080   # the number of lidar points
IMG_SIZE = 64
SEQ_LEN = 10
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, file_name):
        # initialize the data and labels
        # read the names of image data:
        self.scan_file_names = []
        self.pos_file_names = []
        self.vel_file_names = []
        # open train.txt or dev.txt:
        fp_scan = open(img_path+'/scans/'+file_name+'.txt', 'r')
        fp_pos = open(img_path+'/positions/'+file_name+'.txt', 'r')
        fp_vel = open(img_path+'/velocities/'+file_name+'.txt', 'r')
        # for each line of the file:
        for line in fp_scan.read().split(NEW_LINE):
            if('.npy' in line): 
                self.scan_file_names.append(img_path+'/scans/'+line)
        for line in fp_pos.read().split(NEW_LINE):
            if('.npy' in line): 
                self.pos_file_names.append(img_path+'/positions/'+line)
        for line in fp_vel.read().split(NEW_LINE):
            if('.npy' in line): 
                self.vel_file_names.append(img_path+'/velocities/'+line)
        # close txt file:
        fp_scan.close()
        fp_pos.close()
        fp_vel.close()
        self.length = len(self.scan_file_names)

        print("dataset length: ", self.length)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # get the index of start point:
        scans = np.zeros((SEQ_LEN+SEQ_LEN, POINTS))
        positions = np.zeros((SEQ_LEN+SEQ_LEN, 3))
        vels = np.zeros((SEQ_LEN+SEQ_LEN, 2))
        # get the index of start point:
        if(idx+(SEQ_LEN+SEQ_LEN) < self.length): # train1:
            idx_s = idx
        else:
            idx_s = idx - (SEQ_LEN+SEQ_LEN)

        for i in range(SEQ_LEN+SEQ_LEN):
            # get the scan data:
            scan_name = self.scan_file_names[idx_s+i]
            scan = np.load(scan_name)
            scans[i] = scan
            # get the scan_ur data:
            pos_name = self.pos_file_names[idx_s+i]
            pos = np.load(pos_name)
            positions[i] = pos
            # get the velocity data:
            vel_name = self.vel_file_names[idx_s+i]
            vel = np.load(vel_name)
            vels[i] = vel
        
        # initialize:
        scans[np.isnan(scans)] = 20.
        scans[np.isinf(scans)] = 20.
        scans[scans==30] = 20.

        positions[np.isnan(positions)] = 0.
        positions[np.isinf(positions)] = 0.

        vels[np.isnan(vels)] = 0.
        vels[np.isinf(vels)] = 0.

        # transfer to pytorch tensor:
        scan_tensor = torch.FloatTensor(scans)
        pose_tensor = torch.FloatTensor(positions)
        vel_tensor =  torch.FloatTensor(vels)

        data = {
                'scan': scan_tensor,
                'position': pose_tensor,
                'velocity': vel_tensor, 
                }

        return data
