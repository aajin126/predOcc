#!/usr/bin/env python
#
# file: $ISIP_EXP/predOcc/scripts/train.py
#
# revision history: xzt
#  20220824 (TE): first version
#
# usage:
#  python train.py mdir trian_data val_data
#
# arguments:
#  mdir: the directory where the output model is stored
#  trian_data: the directory of training data
#  val_data: the directory of valiation data
#
# This script trains a predOcc model
#------------------------------------------------------------------------------

# import pytorch modules
#
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import wandb

# visualize:
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
# import the model and all of its variables/functions
#
from model import *
from local_occ_grid_map import LocalMap
from util import *

# import modules
#
import sys
import os


#-----------------------------------------------------------------------------
#
# global variables are listed here
#
#-----------------------------------------------------------------------------

# general global values
#
model_dir = './model/model.pth'  # the path of model storage 
NUM_ARGS = 3
IMG_SIZE = 64
NUM_EPOCHS = 50 #100
BATCH_SIZE = 128 #512 #64
LEARNING_RATE = "lr"
BETAS = "betas"
EPS = "eps"
WEIGHT_DECAY = "weight_decay"

# Constants
NUM_INPUT_CHANNELS = 1
NUM_LATENT_DIM = 512 # 16*16*2 
NUM_OUTPUT_CHANNELS = 1
BETA = 0.01
GAMMA = 0.9

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

# adjust_learning_rate
#　
def adjust_learning_rate(optimizer, epoch):
    lr = 1e-4
    if epoch > 30000:
        lr = 3e-4
    if epoch > 50000:
        lr = 2e-5
    if epoch > 48000:
       # lr = 5e-8
       lr = lr * (0.1 ** (epoch // 110000))
    #  if epoch > 8300:
    #      lr = 1e-9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# train function:
def train(model, dataloader, dataset, device, optimizer, criterion, epoch, epochs):
    # set model to training mode:
    model.train()
    # for each batch in increments of batch size:
    running_loss = 0.0
    # kl_divergence:
    kl_avg_loss = 0.0
    # CE loss:
    ce_avg_loss = 0.0
    ce_loss = 0.0
    w_sum = 0.0

    counter = 0
    # get the number of batches (ceiling of train_data/batch_size):
    num_batches = int(len(dataset)/dataloader.batch_size)
    for i, batch in tqdm(enumerate(dataloader), total=num_batches):
    #for i, batch in enumerate(dataloader, 0):
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
        distances = scans[:,SEQ_LEN:] # get future 10 frames
        # the angles of lidar scan: -135 ~ 135 degree
        angles = torch.linspace(-(135*np.pi/180), 135*np.pi/180, distances.shape[-1]).to(device)
        # Lidar measurements in X-Y plane: transform to the predicted robot reference frame
        distances_x, distances_y = mask_gridMap.lidar_scan_xy(distances, angles, x_odom, y_odom, theta_odom)
        # discretize to binary maps:
        mask_binary_maps = mask_gridMap.discretize(distances_x, distances_y)
        # current position:
        obs_pos_N = positions[:, SEQ_LEN-1]
        # calculate relative future positions to current position:
        future_poses = positions[:, SEQ_LEN:] 
        x_rel, y_rel, th_rel = mask_gridMap.robot_coordinate_transform(future_poses, obs_pos_N)


        prediction_maps = torch.zeros(batch_size, SEQ_LEN, 1, IMG_SIZE, IMG_SIZE).to(device)
        prediction_maps_org = torch.zeros(batch_size, SEQ_LEN, 1, IMG_SIZE, IMG_SIZE).to(device)
        # Create input grid maps: 
        input_gridMap = LocalMap(X_lim = MAP_X_LIMIT, 
                    Y_lim = MAP_Y_LIMIT, 
                    resolution = RESOLUTION, 
                    p = P_prior,
                    size=[batch_size, SEQ_LEN],
                    device = device)
        # robot positions:
        pos = positions[:,:SEQ_LEN]
        # Transform the robot past poses to the predicted reference frame.
        x_odom, y_odom, theta_odom =  input_gridMap.robot_coordinate_transform(pos, obs_pos_N)
        # Lidar measurements:
        distances = scans[:,:SEQ_LEN]
        # the angles of lidar scan: -135 ~ 135 degree
        angles = torch.linspace(-(135*np.pi/180), 135*np.pi/180, distances.shape[-1]).to(device)
        # Lidar measurements in X-Y plane: transform to the predicted robot reference frame
        distances_x, distances_y = input_gridMap.lidar_scan_xy(distances, angles, x_odom, y_odom, theta_odom)
        # discretize to binary maps:
        input_binary_maps = input_gridMap.discretize(distances_x, distances_y)
        # occupancy map update:
        input_gridMap.update(x_odom, y_odom, distances_x, distances_y, P_free, P_occ)
        input_occ_grid_map = input_gridMap.to_prob_occ_map(TRESHOLD_P_OCC)

        # add channel dimension:
        input_binary_maps = input_binary_maps.unsqueeze(2)
        mask_binary_maps = mask_binary_maps.unsqueeze(2)

        inputs_samples = input_binary_maps
        inputs_occ_map_samples = input_occ_grid_map

        # set all gradients to 0:
        optimizer.zero_grad()
        # # feed the batch to the network:
        # prediction, kl_loss = model(input_binary_maps, input_occ_grid_map)

        # # warp the prediction(based on current frame) to the target frame(based on t+1 frame):
        # fin_pred_map, valid_mask = reprojection(prediction, x_rel[:,0], y_rel[:,0], th_rel[:,0], MAP_X_LIMIT, MAP_Y_LIMIT) 
        # valid_mask = valid_mask.float()

        for k in range(SEQ_LEN):  
            prediction, kl_loss = model(inputs_samples, inputs_occ_map_samples)
            prediction_t, _ = reprojection(prediction, x_rel[:, k], y_rel[:, k], th_rel[:, k], MAP_X_LIMIT, MAP_Y_LIMIT)
            prediction = prediction.unsqueeze(1)
            prediction_t = prediction_t.unsqueeze(1)
            inputs_samples = torch.cat([inputs_samples[:,1:], prediction], dim=1)

            prediction_maps[:, k] = prediction_t.squeeze(1)
            prediction_maps_org[:, k] = prediction.squeeze(1)
        
        # calculate the total loss:
        for k in range(SEQ_LEN):
            w = (GAMMA ** k)

            # pred/gt: [B,1,H,W]
            pred_k = prediction_maps[:, k]
            gt_k   = mask_binary_maps[:, k]

            loss_k = criterion(pred_k, gt_k).div(batch_size)

            ce_loss = ce_loss + w * loss_k
            w_sum = w_sum + w
        
        ce_loss = ce_loss / w_sum

        if i == 0:
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
            fig.tight_layout()
            wandb.log({"viz/GT": wandb.Image(fig, caption=f"iter={i}")})
            plt.close(fig)

            fig = plt.figure(figsize=(8, 1))
            for m in range(SEQ_LEN):   
                # display the mask of occupancy grids:
                a = fig.add_subplot(1,SEQ_LEN,m+1)
                pred = prediction_maps[0,m]
                input_grid = make_grid(pred.detach().cpu())
                input_image = input_grid.permute(1, 2, 0)
                plt.imshow(input_image)
                plt.xticks([])
                plt.yticks([])
                input_title = "n=" + str(m+1)
                a.set_title(input_title, fontdict={'fontsize': fontsize})
            fig.tight_layout()
            wandb.log({"viz/pred": wandb.Image(fig, caption=f"iter={i}")})
            plt.close(fig)

        # B, _, H, W = fin_pred_map.shape
        # # the number of valid pixels per sample:
        # #valid_sum = valid_mask.flatten(1).sum(1)
        # #valid_ratio = valid_sum.mean()/(H*W)
        # #print(f"Batch {i}: valid ratio: {valid_ratio:.4f}")
        # # calculate the total loss:
        # ce_loss = criterion(fin_pred_map, mask_binary_maps[:,0]).div(batch_size)
        # total loss:
        loss = ce_loss + BETA*kl_loss
        # perform back propagation:
        loss.backward(torch.ones_like(loss))
        optimizer.step()
        # get the loss:
        # multiple GPUs:
        if torch.cuda.device_count() > 1:
            loss = loss.mean()  
            ce_loss = ce_loss.mean()
            kl_loss = kl_loss.mean()

        running_loss += loss.item()
        # kl_divergence:
        kl_avg_loss += kl_loss.item()
        # CE loss:
        ce_avg_loss += ce_loss.item()

        # display informational message:
        if(i % 128 == 0):
            print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, CE_Loss: {:.4f}, KL_Loss: {:.4f}'
                    .format(epoch, epochs, i + 1, num_batches, loss.item(), ce_loss.item(), kl_loss.item()))
    train_loss = running_loss / counter 
    train_kl_loss = kl_avg_loss / counter
    train_ce_loss = ce_avg_loss / counter

    return train_loss, train_kl_loss, train_ce_loss

# validate function:
def validate(model, dataloader, dataset, device, criterion):
    # set model to evaluation mode:
    model.eval()
    # for each batch in increments of batch size:
    running_loss = 0.0
    # kl_divergence:
    kl_avg_loss = 0.0
    # CE loss:
    ce_avg_loss = 0.0
    ce_loss = 0.0
    w_sum = 0.0

    counter = 0
    # get the number of batches (ceiling of train_data/batch_size):
    num_batches = int(len(dataset)/dataloader.batch_size)
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=num_batches):
        #for i, batch in enumerate(dataloader, 0):
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
            # current position:
            obs_pos_N = positions[:, SEQ_LEN-1]
            # calculate relative future positions to current position:
            future_poses = positions[:, SEQ_LEN:] 
            x_rel, y_rel, th_rel = mask_gridMap.robot_coordinate_transform(future_poses, obs_pos_N)

            prediction_maps = torch.zeros(batch_size, SEQ_LEN, 1, IMG_SIZE, IMG_SIZE).to(device)
            prediction_maps_org = torch.zeros(batch_size, SEQ_LEN, 1, IMG_SIZE, IMG_SIZE).to(device)

            # Create input grid maps: 
            input_gridMap = LocalMap(X_lim = MAP_X_LIMIT, 
                        Y_lim = MAP_Y_LIMIT, 
                        resolution = RESOLUTION, 
                        p = P_prior,
                        size=[batch_size, SEQ_LEN],
                        device = device)
            # robot positions:
            pos = positions[:,:SEQ_LEN]
            # Transform the robot past poses to the predicted reference frame.
            x_odom, y_odom, theta_odom =  input_gridMap.robot_coordinate_transform(pos, obs_pos_N)
            # Lidar measurements:
            distances = scans[:,:SEQ_LEN]
            # the angles of lidar scan: -135 ~ 135 degree
            angles = torch.linspace(-(135*np.pi/180), 135*np.pi/180, distances.shape[-1]).to(device)
            # Lidar measurements in X-Y plane: transform to the predicted robot reference frame
            distances_x, distances_y = input_gridMap.lidar_scan_xy(distances, angles, x_odom, y_odom, theta_odom)
            # discretize to binary maps:
            input_binary_maps = input_gridMap.discretize(distances_x, distances_y)
            # occupancy map update:
            input_gridMap.update(x_odom, y_odom, distances_x, distances_y, P_free, P_occ)
            input_occ_grid_map = input_gridMap.to_prob_occ_map(TRESHOLD_P_OCC)

            # add channel dimension:
            input_binary_maps = input_binary_maps.unsqueeze(2)
            mask_binary_maps = mask_binary_maps.unsqueeze(2)
            inputs_samples = input_binary_maps
            inputs_occ_map_samples = input_occ_grid_map
            for k in range(SEQ_LEN):  
                prediction, kl_loss = model(inputs_samples, inputs_occ_map_samples)
                prediction_t, _ = reprojection(prediction, x_rel[:, k], y_rel[:, k], th_rel[:, k], MAP_X_LIMIT, MAP_Y_LIMIT)
                prediction = prediction.unsqueeze(1)
                prediction_t = prediction_t.unsqueeze(1)
                inputs_samples = torch.cat([inputs_samples[:,1:], prediction], dim=1)

                prediction_maps[:, k] = prediction_t.squeeze(1)
                prediction_maps_org[:, k] = prediction.squeeze(1)
            
            # calculate the total loss:
            for k in range(SEQ_LEN):
                w = (GAMMA ** k)

                # pred/gt: [B,1,H,W]
                pred_k = prediction_maps[:, k]
                gt_k   = mask_binary_maps[:, k]

                loss_k = criterion(pred_k, gt_k).div(batch_size)

                ce_loss = ce_loss + w * loss_k
                w_sum = w_sum + w
            
            ce_loss = ce_loss / w_sum

            # # feed the batch to the network:
            # prediction, kl_loss= model(input_binary_maps, input_occ_grid_map)
            
            # # warp the prediction(based on current frame) to the target frame(based on t+1 frame):
            # fin_pred_map, valid_mask = reprojection(prediction, x_rel[:,0], y_rel[:,0], th_rel[:,0], MAP_X_LIMIT, MAP_Y_LIMIT) 
            # valid_mask = valid_mask.float()

            # # calculate the total loss:
            # ce_loss = criterion(fin_pred_map, mask_binary_maps[:,0]).div(batch_size)
            # total loss:
            loss = ce_loss + BETA*kl_loss

            # multiple GPUs:
            if torch.cuda.device_count() > 1:
                loss = loss.mean()
                ce_loss = ce_loss.mean()
                kl_loss = kl_loss.mean()

            # get the loss:
            running_loss += loss.item()
            # kl_divergence:
            kl_avg_loss += kl_loss.item()
            # CE loss:
            ce_avg_loss += ce_loss.item()

    val_loss = running_loss / counter
    val_kl_loss = kl_avg_loss / counter
    val_ce_loss = ce_avg_loss / counter

    return val_loss, val_kl_loss, val_ce_loss

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
    # ensure we have the correct amount of arguments:
    #global cur_batch_win
    if(len(argv) != NUM_ARGS):
        print("usage: python train.py [MDL_PATH] [TRAIN_PATH] [VAL_PATH]")
        exit(-1)

    # define local variables:
    mdl_path = argv[0]
    pTrain = argv[1]
    pDev = argv[2]

    wandb.init(
        project="predOcc",
        name=f"run_{os.path.basename(mdl_path)}",
        config={
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "latent_dim": NUM_LATENT_DIM,
            "beta": BETA,
            "resolution": RESOLUTION,
        }
    )

    # get the output directory name:
    odir = os.path.dirname(mdl_path)

    # if the odir doesn't exits, we make it:
    if not os.path.exists(odir):
        os.makedirs(odir)

    # set the device to use GPU if available:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('...Start reading data...')

    ### training data ###
    # training set and training data loader
    train_dataset = VaeTestDataset(pTrain, 'train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, \
                                                   shuffle=True, drop_last=True, pin_memory=True)

    ### validation data ###
    # validation set and validation data loader
    dev_dataset = VaeTestDataset(pDev, 'val')
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=BATCH_SIZE, num_workers=2, \
                                                 shuffle=True, drop_last=True, pin_memory=True)

    # instantiate a model:
    model = RVAEP(input_channels=NUM_INPUT_CHANNELS,
                  latent_dim=NUM_LATENT_DIM,
                  output_channels=NUM_OUTPUT_CHANNELS)
    # moves the model to device (cpu in our case so no change):
    model.to(device)

    # set the adam optimizer parameters:
    opt_params = { LEARNING_RATE: 0.001,
                   BETAS: (.9,0.999),
                   EPS: 1e-08,
                   WEIGHT_DECAY: .001 }
    # set the loss criterion and optimizer:
    criterion = nn.BCELoss(reduction='sum') 
    criterion.to(device)
    # create an optimizer, and pass the model params to it:
    optimizer = Adam(model.parameters(), **opt_params)

    # get the number of epochs to train on:
    epochs = NUM_EPOCHS

    # if there are trained models, continue training:
    if os.path.exists(mdl_path):
        checkpoint = torch.load(mdl_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('Load epoch {} success'.format(start_epoch))
    else:
        start_epoch = 0
        print('No trained models, restart training')

    # multiple GPUs:
    if torch.cuda.device_count() > 1:
        print("Let's use 2 of total", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model) #, device_ids=[0, 1])
    # moves the model to device (cpu in our case so no change):
    model.to(device)

    # tensorboard writer:
    writer = SummaryWriter('runs')

    # ---------------- BEST checkpoint tracking ----------------
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    best_val_epoch = -1
    best_train_epoch = -1

    best_val_path = os.path.join(odir, "best_val.pth")
    best_train_path = os.path.join(odir, "best_train.pth")

    epoch_num = 0
    for epoch in range(start_epoch+1, epochs):
        # adjust learning rate:
        adjust_learning_rate(optimizer, epoch)
        ################################## Train #####################################
        # for each batch in increments of batch size
        #
        train_epoch_loss, train_kl_epoch_loss, train_ce_epoch_loss = train(
            model, train_dataloader, train_dataset, device, optimizer, criterion, epoch, epochs
        )
        valid_epoch_loss, valid_kl_epoch_loss, valid_ce_epoch_loss = validate(
            model, dev_dataloader, dev_dataset, device, criterion
        )
        wandb.log({
            "train/loss": train_epoch_loss,
            "train/kl_loss": train_kl_epoch_loss,
            "train/ce_loss": train_ce_epoch_loss,
            "val/loss": valid_epoch_loss,
            "val/kl_loss": valid_kl_epoch_loss,
            "val/ce_loss": valid_ce_epoch_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }, step=epoch)

        # log the epoch loss
        writer.add_scalar('training loss',
                        train_epoch_loss,
                        epoch)
        writer.add_scalar('training kl loss',
                        train_kl_epoch_loss,
                        epoch)
        writer.add_scalar('training ce loss',
                train_ce_epoch_loss,
                epoch)
        writer.add_scalar('validation loss',
                        valid_epoch_loss,
                        epoch)
        writer.add_scalar('validation kl loss',
                        valid_kl_epoch_loss,
                        epoch)
        writer.add_scalar('validation ce loss',
                        valid_ce_epoch_loss,
                        epoch)

        print('Train set: Average loss: {:.4f}'.format(train_epoch_loss))
        print('Validation set: Average loss: {:.4f}'.format(valid_epoch_loss))
        
        # save the model:
        if((epoch+1) % 10 == 0):
            if torch.cuda.device_count() > 1: # multiple GPUS: 
                state = {'model':model.module.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            else:
                state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            path='./model/model' + str(epoch) +'.pth'
            torch.save(state, path)

            # ---- wandb checkpoint upload every 10 epochs ----
            artifact = wandb.Artifact(
                name=f"predOcc-{os.path.basename(mdl_path)}-ep{epoch}",
                type="checkpoint",
                metadata={"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
            )
            artifact.add_file(path)
            wandb.log_artifact(artifact)
        
        if torch.cuda.device_count() > 1:
            best_state = {
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": float(train_epoch_loss),
                "val_loss": float(valid_epoch_loss),
            }
        else:
            best_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": float(train_epoch_loss),
                "val_loss": float(valid_epoch_loss),
            }

        # best by val loss
        if valid_epoch_loss < best_val_loss:
            best_val_loss = float(valid_epoch_loss)
            best_val_epoch = int(epoch)
            torch.save(best_state, best_val_path)

            artifact = wandb.Artifact(
                name=f"predOcc-best-val-{os.path.basename(mdl_path)}",
                type="checkpoint",
                metadata={"best_epoch": best_val_epoch, "best_val_loss": best_val_loss, "lr": optimizer.param_groups[0]["lr"]},
            )
            artifact.add_file(best_val_path)
            wandb.log_artifact(artifact)

        # best by train loss
        if train_epoch_loss < best_train_loss:
            best_train_loss = float(train_epoch_loss)
            best_train_epoch = int(epoch)
            torch.save(best_state, best_train_path)

            artifact = wandb.Artifact(
                name=f"predOcc-best-train-{os.path.basename(mdl_path)}",
                type="checkpoint",
                metadata={"best_epoch": best_train_epoch, "best_train_loss": best_train_loss, "lr": optimizer.param_groups[0]["lr"]},
            )
            artifact.add_file(best_train_path)
            wandb.log_artifact(artifact)
        # -------------------------------
        epoch_num = epoch

    # save the final model
    if torch.cuda.device_count() > 1: # multiple GPUS: 
        state = {'model':model.module.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch_num}
    else:
        state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch_num}
    torch.save(state, mdl_path)

    # exit gracefully
    #
    wandb.finish()

    return True
#
# end of function


# begin gracefully
#
if __name__ == '__main__':
    main(sys.argv[1:])
#
# end of file
