import torch
from diffusers import DDPMScheduler, DDIMScheduler

class DiffusionWrapper:
    def __init__(self, num_train_timesteps=1000, beta_schedule="linear", prediction_type="epsilon",
                 use_ddim=False, ddim_steps=50):
        self.train_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,   # "epsilon" (eps-pred)
        )
        # inference scheduler (DDIM)
        self.infer_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
        )
        self.use_ddim = use_ddim
        self.ddim_steps = ddim_steps

    # ========== training ==========
    def add_noise(self, z0, eps, t):
        return self.train_scheduler.add_noise(z0, eps, t)

    def predict_x0(self, z_t, eps_hat, t):
        # x0 = (x_t - sqrt(1-a_t)*eps) / sqrt(a_t)
        alphas_cumprod = self.train_scheduler.alphas_cumprod.to(z_t.device)
        a_t = alphas_cumprod[t].view(-1, *([1] * (z_t.dim() - 1)))
        return (z_t - torch.sqrt(1 - a_t) * eps_hat) / torch.sqrt(a_t)

    # ========== inference ==========
    def set_inference_steps(self, num_inference_steps):
        self.infer_scheduler.set_timesteps(num_inference_steps)

    def step(self, eps_hat, t, z_t):
        # returns prev_sample
        return self.infer_scheduler.step(eps_hat, t, z_t).prev_sample