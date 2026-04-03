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
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import time

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


def binarize_occ(x, occ_thr=0.5):
    return x > occ_thr


def compute_iou(pred, gt, occ_thr=0.3):
    pred_occ = binarize_occ(pred, occ_thr)
    gt_occ = binarize_occ(gt, occ_thr)

    inter = (pred_occ & gt_occ).sum().float()
    union = (pred_occ | gt_occ).sum().float()
    return inter / (union + 1e-6)


def compute_f1(pred, gt, occ_thr=0.5, eps=1e-6):
    pred_occ = binarize_occ(pred, occ_thr)
    gt_occ = binarize_occ(gt, occ_thr)

    tp = (pred_occ & gt_occ).sum().float()
    fp = (pred_occ & (~gt_occ)).sum().float()
    fn = ((~pred_occ) & gt_occ).sum().float()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return 2.0 * precision * recall / (precision + recall + eps)


def compute_f2(pred, gt, occ_thr=0.5, eps=1e-6):
    beta = 2.0
    pred_occ = binarize_occ(pred, occ_thr)
    gt_occ = binarize_occ(gt, occ_thr)

    tp = (pred_occ & gt_occ).sum().float()
    fp = (pred_occ & (~gt_occ)).sum().float()
    fn = ((~pred_occ) & gt_occ).sum().float()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    beta2 = beta ** 2
    return (1.0 + beta2) * precision * recall / (beta2 * precision + recall + eps)


def get_metric_fn(metric_name):
    metric_name = metric_name.lower()
    if metric_name == "iou":
        return compute_iou
    if metric_name == "f1":
        return compute_f1
    if metric_name == "f2":
        return compute_f2
    raise ValueError(f"Unsupported metric: {metric_name}")


def compute_topk_time_independent(pred_samples_t, gt_t, metric_fn, occ_thr=0.5):
    scores = []
    for k in range(pred_samples_t.shape[0]):
        scores.append(metric_fn(pred_samples_t[k], gt_t, occ_thr=occ_thr))

    scores_tensor = torch.stack(scores)
    best_score, best_idx = torch.max(scores_tensor, dim=0)
    return best_score, int(best_idx.item()), [float(s.item()) for s in scores_tensor]


def compute_topk_time_consistent(pred_samples, gt_seq, metric_fn, occ_thr=0.5, reduce="mean"):
    sample_scores = []
    per_sample_per_t = []

    for k in range(pred_samples.shape[0]):
        per_t_scores = []
        for t in range(pred_samples.shape[1]):
            per_t_scores.append(metric_fn(pred_samples[k, t], gt_seq[t], occ_thr=occ_thr))

        per_t_tensor = torch.stack(per_t_scores)
        per_sample_per_t.append(per_t_tensor)

        if reduce == "sum":
            agg = per_t_tensor.sum()
        elif reduce == "mean":
            agg = per_t_tensor.mean()
        else:
            raise ValueError(f"Unsupported reduce: {reduce}")

        sample_scores.append(agg)

    sample_scores_tensor = torch.stack(sample_scores)
    best_seq_score, best_sample_idx = torch.max(sample_scores_tensor, dim=0)
    best_per_t = per_sample_per_t[int(best_sample_idx.item())]

    return (
        best_seq_score,
        int(best_sample_idx.item()),
        [float(s.item()) for s in sample_scores_tensor],
        [float(s.item()) for s in best_per_t],
    )


def save_gt_maps(mask_binary_maps, output_dir, batch_idx):
    fontsize = 8
    fig = plt.figure(figsize=(8, 1))
    for m in range(SEQ_LEN):
        axis = fig.add_subplot(1, SEQ_LEN, m + 1)
        mask = mask_binary_maps[0, m].detach().cpu()
        grid = make_grid(mask)
        image = grid.permute(1, 2, 0)
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        axis.set_title(f"n={m+1}", fontdict={"fontsize": fontsize})
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"mask_{batch_idx}.jpg"))
    plt.close(fig)


def save_pred_mean_maps(prediction_maps, output_dir, batch_idx):
    fontsize = 8
    fig = plt.figure(figsize=(8, 1))
    for m in range(SEQ_LEN):
        axis = fig.add_subplot(1, SEQ_LEN, m + 1)
        pred_mean = prediction_maps[:, m].mean(dim=0).detach().cpu()
        grid = make_grid(pred_mean)
        image = grid.permute(1, 2, 0)
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        axis.set_title(f"n={m+1}", fontdict={"fontsize": fontsize})
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"pred_{batch_idx}.jpg"))
    plt.close(fig)


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("mdl_path", type=str, help="Path to model checkpoint")
    parser.add_argument("eval_set", type=str, help="Path to evaluation dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("output", "v1.6"),
        help="Directory where eval_table.csv and images are saved",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="iou",
        choices=["iou", "f1", "f2"],
        help="Metric used for Top-K evaluation",
    )
    parser.add_argument(
        "--occ_thr",
        type=float,
        default=0.3,
        help="Occupancy threshold used by the selected metric",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of stochastic VAE samples per input",
    )
    parser.add_argument(
        "--no_images",
        action="store_true",
        help="Skip saving visualization images",
    )
    return parser.parse_args(argv)

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
    opt = parse_args(argv)
    mdl_path = opt.mdl_path
    fImg = opt.eval_set
    metric_fn = get_metric_fn(opt.metric)
    os.makedirs(opt.output_dir, exist_ok=True)
    csv_path = os.path.join(opt.output_dir, "eval_table.csv")

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
    all_rows = []
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
       
            prediction_maps = torch.zeros(opt.num_samples, SEQ_LEN, 1, IMG_SIZE, IMG_SIZE, device=device)
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
            inputs_samples = input_binary_maps.repeat(opt.num_samples, 1, 1, 1, 1)
            inputs_occ_map_samples = input_occ_grid_map.unsqueeze(1).repeat(opt.num_samples, 1, 1, 1)
            
            # start timing
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            prediction, kl_loss = model(inputs_samples, inputs_occ_map_samples) #(B, T, C, H, W)

            for k in range(SEQ_LEN):  
                prediction_t, _ = reprojection(prediction[:, k], x_rel[:, k], y_rel[:, k], th_rel[:, k], MAP_X_LIMIT, MAP_Y_LIMIT)
                prediction_t = prediction_t.reshape(-1, 1, IMG_SIZE, IMG_SIZE)
                prediction_maps[:, k] = prediction_t

            # end timing
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            total_pred_time_ms = (t1 - t0) * 1000

            row = {"i": int(i), "Inference_time": float(total_pred_time_ms)}
            for n in range(SEQ_LEN):
                gt_map = mask_binary_maps[0, n]
                pred_samples_t = prediction_maps[:, n]
                topk_ind_score, _, _ = compute_topk_time_independent(
                    pred_samples_t=pred_samples_t,
                    gt_t=gt_map,
                    metric_fn=metric_fn,
                    occ_thr=opt.occ_thr,
                )
                row[f"n={n+1}_topk_{opt.metric}_ind"] = float(topk_ind_score.item())

            gt_seq = mask_binary_maps[0]
            _, _, _, best_per_t_scores = compute_topk_time_consistent(
                pred_samples=prediction_maps,
                gt_seq=gt_seq,
                metric_fn=metric_fn,
                occ_thr=opt.occ_thr,
                reduce="mean",
            )

            for n in range(SEQ_LEN):
                row[f"n={n+1}_topk_{opt.metric}_cons"] = float(best_per_t_scores[n])
            all_rows.append(row)

            if (i + 1) % 100 == 0:
                pd.DataFrame(all_rows).to_csv(csv_path, index=False)

            if not opt.no_images:
                save_gt_maps(mask_binary_maps, opt.output_dir, i)
                save_pred_mean_maps(prediction_maps, opt.output_dir, i)

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
            # input_img_name = "./output/pred_org" + str(i)+ ".jpg"
            # plt.savefig(input_img_name)
            # plt.close(fig)

            print(i)

    df = pd.DataFrame(all_rows)
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
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
