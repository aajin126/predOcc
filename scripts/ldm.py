import torch
import torch.nn as nn
import torch.nn.functional as F

from codec import Encoder, Decoder
from ddpm import DiffusionWrapper
from unet import UNetModel

class LatentDiffusionModel(nn.Module):
    def __init__(
        self,
        # codec
        map_in_ch=1,
        map_out_ch=1,
        num_hiddens=64,
        num_res_layers=2,
        num_res_hiddens=32,
        # latent
        z_channels=64,       
        # diffusion unet
        model_channels=64,
        num_res_blocks=2,
        channel_mult=(1,2,4),
        # conditioning
        cond_channels=64,
        # diffusion schedule
        num_train_timesteps=1000,
    ):
        super().__init__()

        # ---------- codec (deterministic AE) ----------
        self.encoder = Encoder(map_in_ch, num_hiddens, num_res_layers, num_res_hiddens)  # (B,1,H,W)->(B,zC,h,w)
        self.decoder = Decoder(map_out_ch, num_hiddens, num_res_layers, num_res_hiddens) # (B,zC,h,w)->(B,1,H,W)

        # ---------- diffusion scheduler wrapper ----------
        self.diff = DiffusionWrapper(num_train_timesteps=num_train_timesteps)

        # ---------- denoiser UNet (latent space) ----------
        # IMPORTANT: in_channels/out_channels should match z_channels
        self.unet = UNetModel(
            in_channels=z_channels,
            model_channels=model_channels,
            out_channels=z_channels,
            num_res_blocks=num_res_blocks,
            channel_mult=channel_mult,
            dims=3,
            use_scale_shift_norm=True,   # FiLM-like in ResBlock 
        )

        self.cond_channels = cond_channels

    def encode_seq(self, Y_gt_t):
        # Y_gt_t: (B,T,1,H,W) -> z0: (B,Cz,T,h,w)
        B, T, C, H, W = Y_gt_t.shape
        z_list = []
        for i in range(T):
            z_i = self.encoder(Y_gt_t[:, i])  # (B,Cz,h,w) 
            z_list.append(z_i)
        z0 = torch.stack(z_list, dim=2)
        return z0

    def decode_seq(self, z):
        # z: (B,Cz,T,h,w) -> Y_hat_t: (B,T,1,H,W)
        B, Cz, T, h, w = z.shape
        y_list = []
        for i in range(T):
            y_i = self.decoder(z[:, :, i])  # (B,1,H,W)  
            y_list.append(y_i)
        Y_hat_t = torch.stack(y_list, dim=1)
        return Y_hat_t

    def forward_train(self, Y_gt_t, cond_z=None, lambda_reproj=0.0, Y_gt_future_frame=None, reproj_fn=None):
        """
        Y_gt_t: (B,T,1,H,W)  # already warped to t-frame
        cond_z: (B,Cc,h,w) or None  # condition (you will inject to UNet; currently unet doesn't accept it yet)
        """
        # 1) target latent
        z0 = self.encode_seq(Y_gt_t)                            # (B,Cz,T,h,w)
        # 2) sample diffusion timestep + noise
        t = torch.randint(0, self.diff.train_scheduler.config.num_train_timesteps, (z0.shape[0],), device=z0.device)
        eps = torch.randn_like(z0)
        # 3) forward noising
        z_t = self.diff.add_noise(z0, eps, t)               

        # 4) noise prediction with condition (FiLM via emb)
        eps_hat = self.unet(z_t, timesteps=t, cond_z=cond_z)     
        L_diff = F.mse_loss(eps_hat, eps)       

        out = {"L_diff": L_diff, "eps_hat": eps_hat}

        out["L_total"] = L_diff
        
        # 5) optional reprojection aux loss
        # if lambda_reproj > 0.0 and reproj_fn is not None and Y_gt_future_frame is not None:
        #     z0_hat = self.diff.predict_x0(z_t, eps_hat, t)     
        #     Y_hat_t = self.decode_seq(z0_hat)
        #     Y_hat_future = reproj_fn(Y_hat_t)                   # 기존 reproj 함수
        #     L_reproj = F.binary_cross_entropy(Y_hat_future, Y_gt_future_frame)
        #     out["L_reproj"] = L_reproj
        #     out["L_total"] = L_diff + lambda_reproj * L_reproj
        # else:
        #     out["L_total"] = L_diff

        return out
