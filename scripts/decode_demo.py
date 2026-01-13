#!/usr/bin/env python
#
# file: $ISIP_EXP/SOGMP/scripts/decode_demo.py
#
# revision history: xzt
#  20220824 (TE): first version
#
# usage:
#  python decode_demo.py mdir mdl_path test_data
#
# arguments:
#  mdir: the directory where the output results are stored
#  mdl_path: the directory of training data
#  test_data: the directory of testing data
#
# This script decodes a SOGMP++ model and gives a result demo
#------------------------------------------------------------------------------

# import pytorch modules
#
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

# visualize:
from tensorboardX import SummaryWriter
#from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

import torchvision.transforms as transforms
import torchvision
import matplotlib
from torchvision.utils import make_grid
#from utils import save_reconstructed_images, image_to_vid, save_loss_plot
matplotlib.style.use('ggplot')

# import modules
#
import sys
import os

# import the model and all of its variables/functions
#
from model import *
from local_occ_grid_map import LocalMap
from transform_util import *
from eval import *

#-----------------------------------------------------------------------------
#
# global variables are listed here
#
#-----------------------------------------------------------------------------

# general global values
#
NUM_ARGS = 2
IMG_SIZE = 64
SPACE = " "        
log_dir = '../model/model.pth'   

# Constants
NUM_CLASSES = 1
NUM_INPUT_CHANNELS = 1
NUM_LATENT_DIM = 512
NUM_OUTPUT_CHANNELS = NUM_CLASSES

# Init map parameters
P_prior = 0.5	# Prior occupancy probability
P_occ = 0.7	    # Probability that cell is occupied with total confidence
P_free = 0.3	# Probability that cell is free with total confidence 
MAP_X_LIMIT = [0, 6.4]      # Map limits on the x-axis
MAP_Y_LIMIT = [-3.2, 3.2]   # Map limits on the y-axis
RESOLUTION = 0.1        # Grid resolution in [m]'
TRESHOLD_P_OCC = 0.8    # Occupancy threshold

# for reproducibility, we seed the rng
#
set_seed(SEED1)        

#------------------------------------------------------------------------------
#
# the main program starts here
#
#------------------------------------------------------------------------------

# function: main
#
# arguments: none
#
# return: none
#
# This method is the main function.
#
def main(argv):
    # ensure we have the correct number of arguments:
    if(len(argv) != NUM_ARGS):
        print("usage: python decode_demo.py [ODIR] [MDL_PATH] [EVAL_SET]")
        exit(-1)

    # define local variables:
    mdl_path = argv[0]
    fImg = argv[1]

    # set the device to use GPU if available:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get array of the data
    # data: [[0, 1, ... 26], [27, 28, ...] ...]
    # labels: [0, 0, 1, ...]
    #
    #[ped_pos_e, scan_e, goal_e, vel_e] = get_data(fname)
    eval_dataset = VaeTestDataset(fImg,'test')
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, \
                                                   shuffle=False, drop_last=True) #, pin_memory=True)

    # instantiate a model:
    model = RVAEP(input_channels=NUM_INPUT_CHANNELS,
                      latent_dim=NUM_LATENT_DIM,
                      output_channels=NUM_OUTPUT_CHANNELS)
    # moves the model to device (cpu in our case so no change):
    model.to(device)

    # set the model to evaluate
    #
    model.eval()

    # set the loss criterion:
    criterion = nn.MSELoss(reduction='sum') #, weight=class_weights)
    criterion.to(device)

    # load the weights
    #
    checkpoint = torch.load(mdl_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # for each batch in increments of batch size:
    counter = 0
    # get the number of batches (ceiling of train_data/batch_size):
    num_batches = int(len(eval_dataset)/eval_dataloader.batch_size)
    with torch.no_grad():
        for i, batch in tqdm(enumerate(eval_dataloader), total=num_batches):
            counter += 1
            # collect the samples as a batch:
            scans = batch['scan']
            scans = scans.to(device)
            positions = batch['position']
            positions = positions.to(device)
            velocities = batch['velocity']
            velocities = velocities.to(device)

            # create occupancy maps:
            batch_size = scans.size(0)
            # Create mask grid maps:
            mask_gridMap = LocalMap(X_lim = MAP_X_LIMIT, 
                            Y_lim = MAP_Y_LIMIT, 
                            resolution = RESOLUTION, 
                            p = P_prior,
                            size=[batch_size, SEQ_LEN],
                            device = device)
            # robot positions:
            x_odom = torch.zeros(batch_size, SEQ_LEN).to(device)
            y_odom = torch.zeros(batch_size, SEQ_LEN).to(device)
            theta_odom = torch.zeros(batch_size, SEQ_LEN).to(device)
            # Lidar measurements:
            distances = scans[:,SEQ_LEN:]
            # the angles of lidar scan: -135 ~ 135 degree
            angles = torch.linspace(-(135*np.pi/180), 135*np.pi/180, distances.shape[-1]).to(device)
            # Lidar measurements in X-Y plane: transform to the predicted robot reference frame
            distances_x, distances_y = mask_gridMap.lidar_scan_xy(distances, angles, x_odom, y_odom, theta_odom)
            # discretize to binary maps:
            mask_binary_maps = mask_gridMap.discretize(distances_x, distances_y)
            mask_binary_maps = mask_binary_maps.unsqueeze(2)

            # current position:
            obs_pos_N = positions[:, SEQ_LEN-1]
            # calculate relative future positions to current position:
            future_poses = positions[:, SEQ_LEN:] 
            x_rel, y_rel, th_rel = mask_gridMap.robot_coordinate_transform(future_poses, obs_pos_N)
       
            prediction_maps = torch.zeros(SEQ_LEN, 1, IMG_SIZE, IMG_SIZE).to(device)
            # multi-step prediction: 10 time steps:

            # Create input grid maps: 
            input_gridMap = LocalMap(X_lim = MAP_X_LIMIT, 
                        Y_lim = MAP_Y_LIMIT, 
                        resolution = RESOLUTION, 
                        p = P_prior,
                        size=[batch_size, SEQ_LEN],
                        device = device)
            pos_origin = positions[:, SEQ_LEN-1]
            # robot positions:
            pos = positions[:,:SEQ_LEN]
            # Transform the robot past poses to the predicted reference frame.
            x_odom, y_odom, theta_odom =  input_gridMap.robot_coordinate_transform(pos, pos_origin)
            # Lidar measurements:
            distances = scans[:,:SEQ_LEN]
            # the angles of lidar scan: -135 ~ 135 degree
            angles = torch.linspace(-(135*np.pi/180), 135*np.pi/180, distances.shape[-1]).to(device)
            # Lidar measurements in X-Y plane: transform to the predicted robot reference frame
            distances_x, distances_y = input_gridMap.lidar_scan_xy(distances, angles, x_odom, y_odom, theta_odom)
            # discretize to binary maps:
            input_binary_maps = input_gridMap.discretize(distances_x, distances_y)
            # local occupancy map update:
            input_gridMap.update(x_odom, y_odom, distances_x, distances_y, P_free, P_occ)
            input_occ_grid_map = input_gridMap.to_prob_occ_map(TRESHOLD_P_OCC)
            # binary occupancy maps:
            input_binary_maps = input_binary_maps.unsqueeze(2)

            # feed the batch to the network:
            num_samples = 32 #1
            inputs_samples = input_binary_maps.repeat(num_samples,1,1,1,1)
            inputs_occ_map_samples = input_occ_grid_map.repeat(num_samples,1,1,1,1)
                
            for t in range(SEQ_LEN):
                prediction, kl_loss = model(inputs_samples, inputs_occ_map_samples)
                prediction = prediction.reshape(-1,1,1,IMG_SIZE,IMG_SIZE)
                inputs_samples = torch.cat([inputs_samples[:,1:], prediction], dim=1)

                predictions = prediction.squeeze(1)

                # mean and std:
                pred_mean = torch.mean(predictions, dim=0, keepdim=True)
                prediction_maps[t, 0] = pred_mean.squeeze()

            fin_prediction_maps = torch.zeros(SEQ_LEN, 1, IMG_SIZE, IMG_SIZE).to(device)
            valid_masks = torch.zeros(SEQ_LEN, 1, IMG_SIZE, IMG_SIZE).to(device)

            for k in range(SEQ_LEN):
                pred_t = prediction_maps[k, 0].unsqueeze(0).unsqueeze(0)   # [1,1,H,W]

                pred_warp, valid_mask = calc_valid_map(
                    pred_t,
                    -x_rel[:, k], -y_rel[:, k], -th_rel[:, k],
                    MAP_X_LIMIT, MAP_Y_LIMIT
                )

                fin_prediction_maps[k, 0] = pred_warp.squeeze(0).squeeze(0)      # [H,W]
                valid_masks[k, 0] = valid_mask.squeeze(0).squeeze(0).float()

            iou_all_steps = []
            iou_valid_steps = []

            for m in range(SEQ_LEN):
                pred = fin_prediction_maps[m, 0]   
                gt   = mask_binary_maps[0, m, 0]
                vmsk = valid_masks[m, 0]

                iou_all = compute_iou(pred, gt, valid_mask=None, thr=0.5)
                iou_v   = compute_iou(pred, gt, valid_mask=vmsk, thr=0.5)

                iou_all_steps.append(iou_all.item())
                iou_valid_steps.append(iou_v.item())

                print(f"[{m+1}] IoU(all)={iou_all.item():.4f}, IoU(valid)={iou_v.item():.4f}")

            # display input occupancy map:
            fig = plt.figure(figsize=(8, 1))
            for m in range(SEQ_LEN):   
                # display the mask of occupancy grids:
                a = fig.add_subplot(1,10,m+1)
                mask = mask_binary_maps[0, m]
                input_grid = make_grid(mask.detach().cpu())
                input_image = input_grid.permute(1, 2, 0)
                plt.imshow(input_image)
                plt.xticks([])
                plt.yticks([])
                fontsize = 8
                input_title = "n=" + str(m+1)
                a.set_title(input_title, fontdict={'fontsize': fontsize})
            input_img_name = "./output/mask" + str(i)+ ".jpg"
            plt.savefig(input_img_name)

            fig = plt.figure(figsize=(8, 1))
            for m in range(SEQ_LEN):   
                # display the mask of occupancy grids:
                a = fig.add_subplot(1,10,m+1)
                pred = fin_prediction_maps[m]
                input_grid = make_grid(pred.detach().cpu())
                input_image = input_grid.permute(1, 2, 0)
                plt.imshow(input_image)
                plt.xticks([])
                plt.yticks([])
                input_title = "n=" + str(m+1)
                a.set_title(input_title, fontdict={'fontsize': fontsize})
            input_img_name = "./output/pred" + str(i)+ ".jpg"
            plt.savefig(input_img_name)
            plt.show()

            print(i)
        
    
    # exit gracefully
    #
    return True
#
# end of function


# begin gracefully
#
if __name__ == '__main__':
    main(sys.argv[1:])
#
# end of file
