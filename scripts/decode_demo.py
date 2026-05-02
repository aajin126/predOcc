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
import time
from PIL import Image
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
import pandas as pd

# import modules
#
import sys
import os

# import the model and all of its variables/functions
#
from model import *
from local_occ_grid_map import LocalMap
from util import *
from eval import *

#-----------------------------------------------------------------------------
#
# global variables are listed here
#
#-----------------------------------------------------------------------------

# general global values
#
NUM_ARGS = 3
IMG_SIZE = 64
SPACE = " "        
log_dir = '../model/model.pth'   
IOU_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

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
all_ssim_rows = []
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
def save_prediction_overlay_gif(prediction_maps, gt_binary, output_path,
                                frame_duration_ms=100, scale=8):
    """Save a GIF of predicted maps with GT occupied cells highlighted in red."""
    frames = []

    for frame_idx in range(prediction_maps.shape[0]):
        pred_map = prediction_maps[frame_idx, 0].detach().cpu().clamp(0, 1).numpy()
        gt_map = gt_binary[frame_idx, 0].detach().cpu().numpy() > 0.5

        pred_uint8 = (pred_map * 255).astype(np.uint8)
        rgb_frame = np.stack([pred_uint8, pred_uint8, pred_uint8], axis=-1)
        rgb_frame[gt_map] = np.array([255, 0, 0], dtype=np.uint8)

        pil_frame = Image.fromarray(rgb_frame, mode="RGB")
        if scale != 1:
            pil_frame = pil_frame.resize(
                (IMG_SIZE * scale, IMG_SIZE * scale),
                Image.Resampling.NEAREST,
            )
        frames.append(pil_frame)

    if frames:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=0,
        )


def save_iou_tables(all_rows_by_thr, output_dir):
    for occ_thr, rows in all_rows_by_thr.items():
        csv_path = os.path.join(output_dir, f"eval_table_iou_{occ_thr:.1f}.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

def save_ssim_table(all_ssim_rows, output_dir):
    """Save SSIM metrics to CSV file."""
    if all_ssim_rows:
        csv_path = os.path.join(output_dir, "eval_table_ssim.csv")
        pd.DataFrame(all_ssim_rows).to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")


def main(argv):
    # ensure we have the correct number of arguments:
    if(len(argv) != NUM_ARGS):
        print("usage: python decode_demo.py [ODIR] [MDL_PATH] [EVAL_SET]")
        exit(-1)

    # define local variables:
    output_dir = argv[0]
    mdl_path = argv[1]
    fImg = argv[2]
    os.makedirs(output_dir, exist_ok=True)

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
    all_rows_by_thr = {occ_thr: [] for occ_thr in IOU_THRESHOLDS}
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
            prediction_maps_org = torch.zeros(SEQ_LEN, 1, IMG_SIZE, IMG_SIZE).to(device)
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
            num_samples = 1
            inputs_samples = input_binary_maps.repeat(num_samples,1,1,1,1)
            inputs_occ_map_samples = input_occ_grid_map.repeat(num_samples,1,1,1,1)
            
            # start timing
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            prediction, kl_loss = model(inputs_samples, inputs_occ_map_samples) #(B, T, C, H, W)

            for k in range(SEQ_LEN):  
                prediction_t, _ = reprojection(prediction[:, k], x_rel[:, k], y_rel[:, k], th_rel[:, k], MAP_X_LIMIT, MAP_Y_LIMIT)
                prediction_t = prediction_t.reshape(-1,1,1,IMG_SIZE,IMG_SIZE)
                predictions = prediction_t.squeeze(1) 
                pred_mean = torch.mean(predictions, dim=0, keepdim=True)
                prediction_maps[k, 0] = pred_mean.squeeze()

            # end timing
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            total_pred_time_ms = (t1 - t0) * 1000

            ssim_row = {
                "i": int(i),
            }
            for n in range(SEQ_LEN):
                gt_map = mask_binary_maps[0, n]
                pred_map = prediction_maps[n]
                ssim_value = compute_ssim_metric(pred_map, gt_map)
                ssim_row[f"n={n+1}"] = ssim_value
            all_ssim_rows.append(ssim_row)

            for occ_thr in IOU_THRESHOLDS:
                row = {
                    "i": int(i),
                    "Inference_time": float(total_pred_time_ms),
                    "occ_thr": float(occ_thr),
                }
                for n in range(SEQ_LEN):
                    gt_map = mask_binary_maps[0, n]
                    pred_map = prediction_maps[n]
                    iou_value = float(compute_iou(pred_map, gt_map, occ_thr=occ_thr).item())
                    row[f"n={n+1}"] = iou_value
                all_rows_by_thr[occ_thr].append(row)

            if (i + 1) % 100 == 0:
                #save_iou_tables(all_rows_by_thr, output_dir)
                save_ssim_table(all_ssim_rows, output_dir)

            # display input occupancy map:
            # fig = plt.figure(figsize=(8, 1))
            # for m in range(SEQ_LEN):   
            #     # display the mask of occupancy grids:
            #     a = fig.add_subplot(1,10,m+1)
            #     mask = mask_binary_maps[0, m]
            #     input_grid = make_grid(mask.detach().cpu())
            #     input_image = input_grid.permute(1, 2, 0)
            #     plt.imshow(input_image)
            #     plt.xticks([])
            #     plt.yticks([])
            #     fontsize = 8
            #     input_title = "n=" + str(m+1)
            #     a.set_title(input_title, fontdict={'fontsize': fontsize})
            # input_img_name = os.path.join(output_dir, "mask" + str(i) + ".jpg")
            # plt.savefig(input_img_name)
            # plt.close(fig)

            # fig = plt.figure(figsize=(8, 1))
            # for m in range(SEQ_LEN):   
            #     # display the mask of occupancy grids:
            #     a = fig.add_subplot(1,10,m+1)
            #     pred = prediction_maps[m]
            #     input_grid = make_grid(pred.detach().cpu())
            #     input_image = input_grid.permute(1, 2, 0)
            #     plt.imshow(input_image)
            #     plt.xticks([])
            #     plt.yticks([])
            #     input_title = "n=" + str(m+1)
            #     a.set_title(input_title, fontdict={'fontsize': fontsize})
            # input_img_name = os.path.join(output_dir, "pred" + str(i) + ".jpg")
            # plt.savefig(input_img_name)
            # plt.close(fig)

            # overlay_gif_name = os.path.join(output_dir, "pred_gif" + str(i) + ".gif")
            # save_prediction_overlay_gif(prediction_maps, mask_binary_maps[0], overlay_gif_name)

            # fig = plt.figure(figsize=(8, 1))
            # for m in range(SEQ_LEN):   
            #     # display the mask of occupancy grids:
            #     a = fig.add_subplot(1,10,m+1)
            #     pred = prediction_maps_org[m]
            #     input_grid = make_grid(pred.detach().cpu())
            #     input_image = input_grid.permute(1, 2, 0)
            #     plt.imshow(input_image)
            #     plt.xticks([])
            #     plt.yticks([])
            #     input_title = "n=" + str(m+1)
            #     a.set_title(input_title, fontdict={'fontsize': fontsize})
            # input_img_name = os.path.join(output_dir, "pred_org" + str(i) + ".jpg")
            # plt.savefig(input_img_name)
            # plt.close(fig)

            print(i)

    save_iou_tables(all_rows_by_thr, output_dir)
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
