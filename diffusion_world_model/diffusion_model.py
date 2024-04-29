# Author: Vibhakar Mohta vmohta@cs.cmu.edu
#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diffusion_model_utils import *
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from vae import VAE
import cv2

class DiffusionTrainer(nn.Module):

    def __init__(self,
                config,
            ):

        super().__init__()
        self.config = config
        
        # VAE
        self.vae = VAE(config)
        checkpoint = torch.load(config["vae_pretrained_ckeckpoint"])
        self.vae.load_state_dict(checkpoint['state_dict'])
        for param in self.vae.parameters():
            param.requires_grad = False
        print("Loaded VAE weights from: ", config["vae_pretrained_ckeckpoint"])
        
        # create network objects
        cond_dim = config["world_action_classes"]
        cond_dim = cond_dim + config["latent_dim"] if config["condition_on_latent"] else cond_dim
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=config["latent_dim"],
            global_cond_dim=cond_dim
        )
        self.noise_pred_net_eval = ConditionalUnet1D(
            input_dim=config["latent_dim"],
            global_cond_dim=cond_dim
        )
        print("Condition Dim: ", cond_dim)
        self.nets = nn.ModuleDict({
            'noise_pred_net': self.noise_pred_net
        })
        self.inference_nets = nn.ModuleDict({
            'noise_pred_net': self.noise_pred_net_eval
        })

        # for this demo, we use DDPMScheduler with 100 diffusion iterations
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.config["num_diffusion_iters"],
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type='epsilon'
        )
        
        self.ddim_sampler = DDIMScheduler(
            num_train_timesteps=self.config["num_ddim_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
        
        # put network on device
        self.device = config["device"]
        self.vae.to(self.device)
        self.nets.to(self.device)
        self.inference_nets.to(self.device)

        # Exponential Moving Average accelerates training and improves stability, holds a copy of the model weights
        self.ema = EMAModel(
            parameters=self.nets.parameters(),
            decay=0.75)

        self.optimizer = torch.optim.AdamW(params=self.nets.parameters(), lr=self.config["lr"], weight_decay=1e-6)

        # Cosine LR schedule with linear warmup
        self.lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=self.optimizer,
            num_warmup_steps=500,
            # num_training_steps= self.num_batches* self.num_epochs
            num_training_steps= self.config["num_epochs"]*self.config["num_batches"]
        )

    def eval(self):
        # self.nets.eval()
        self.inference_nets.eval()
            
    def get_conditioning(self, obs: torch.Tensor, action: torch.Tensor):    
        # action shape: (B, 1)
        action = F.one_hot(action.long(), num_classes=self.config["world_action_classes"]).float()    # (B, 1, self.config["world_action_classes"])
        if self.config["condition_on_latent"] == False:
            return action.squeeze(1)
        else: 
            action = action.squeeze(1)
            return torch.cat([obs.squeeze(1), action], dim=-1).flatten(start_dim=1) # (B, latent_dim + action_dim)
        
    def get_pred(self, obs: torch.Tensor, action: torch.Tensor, sampler = "ddim"):
        """
        Sampler: either ddpm or ddim
        Returns sample: (B, 1, latent_dim)
        """
        if sampler not in ["ddpm", "ddim"]:
            print("Sampler must be either ddpm or ddim")
            return None
        
        with torch.no_grad():
            B = obs.shape[0]
             
            obs_latent = self.vae.get_encoding(obs).unsqueeze(1).detach() # (B, 1, latent_dim)
            noisy_latent = torch.randn_like(obs_latent).to(self.device) # (B, C, H, W)
            cond = self.get_conditioning(obs_latent, action)
                
            if sampler == "ddpm":
                eval_sampler = self.noise_scheduler
                eval_itrs = self.config["num_diffusion_iters"]
            if sampler == "ddim":
                eval_sampler = self.ddim_sampler
                eval_itrs = self.config["num_ddim_iters"]
            
            # init scheduler
            eval_sampler.set_timesteps(eval_itrs)

            # initialize action from obs_latent
            sample = noisy_latent
            # sample
            for k in eval_sampler.timesteps:
                # predict noise
                noise_pred = self.nets['noise_pred_net']( # TODO: change to inference_nets later
                    sample=sample,
                    timestep=k,
                    global_cond=cond
                )

                # inverse diffusion step (remove noise)
                sample = eval_sampler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=sample
                ).prev_sample
            
            # get predicted next_obs_latent
            next_obs_latent = sample + obs_latent
            return next_obs_latent
                
    def train_model_step(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, step=None):
        """
        Input: nimage, nagent_pos, naction in the dataset [normalized inputs]
        Train the model for one step
        Returns the loss        
        """
        B = obs.shape[0]
        
        # Get encoded features
        obs_latent = self.vae.get_encoding(obs).unsqueeze(1).detach() # (B, 1, latent_dim)
        next_obs_latent = self.vae.get_encoding(next_obs).unsqueeze(1).detach() # (B, 1, latent_dim)
        cond = self.get_conditioning(obs_latent, action) # (B, action_dim)
        
        # Initialize random noise
        noise = torch.randn_like(obs_latent) # (B, 1, latent_dim)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=self.device
        ).long()
        
        # Forward diffusion step: Add noise gt_residual to intepolate between gt_residual and gaussian noise
        gt_diff = (next_obs_latent - obs_latent).detach()
        gt_diff_noisy = self.noise_scheduler.add_noise(gt_diff, noise, timesteps) 
        
        # Predict back the noise
        noise_pred = self.nets['noise_pred_net'](gt_diff_noisy, timesteps, global_cond=cond)

        # L2 loss
        loss = F.mse_loss(noise_pred, noise)

        # optimize
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # step lr scheduler every batch, this is different from standard pytorch behavior
        self.lr_scheduler.step()

        # update Exponential Moving Average of the model weights
        # PASS all model parameters
        self.ema.step(self.nets.parameters())

        return loss.item()
    
    def run_after_epoch(self):
        self.ema.copy_to(self.inference_nets.parameters())
    
    def eval_model(self, step, obs, action, next_obs, sampler="ddim", save = False):
        """
        Compute 2 losses:
        1. Latent space loss: MSE between predicted and actual latent space observation
        2. Reconstruction loss: MSE between predicted and actual decoded observation
        """
        with torch.no_grad():
            next_obs_latent = self.vae.get_encoding(next_obs).unsqueeze(1) # (B, 1, latent_dim) GT next latent space
            
            # Get model prediction
            model_pred = self.get_pred(obs, action, sampler)
            
            # Compute the latent space loss
            latent_loss = F.mse_loss(model_pred, next_obs_latent)
            
            # Decode the predicted latent space
            decoded_model_pred = self.vae.decoder(model_pred.squeeze(1)) # (B, C, H, W)
            
            stacked = None
            if save:
                grid1 = torchvision.utils.make_grid(decoded_model_pred[0:5], normalize=True, nrow=5)
                grid2 = torchvision.utils.make_grid(next_obs[0:5], normalize=True, nrow=5)
                stacked = torch.cat([grid1, grid2], dim=1)
            
            recon_loss = F.mse_loss(decoded_model_pred, next_obs)
            
            losses = {
                "latent_loss": latent_loss.item(),
                "recon_loss": recon_loss.item()
            }
            return losses, stacked

    def put_network_on_device(self):
        self.nets.to(self.device)
        self.inference_nets.to(self.device)
        self.vae.to(self.device)
                    
    def load_model_weights(self, model_weights):
        """
        Load the model weights
        """
        self.put_network_on_device()
        self.load_state_dict(model_weights)