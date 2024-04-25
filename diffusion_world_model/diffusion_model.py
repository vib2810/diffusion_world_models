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
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=config["latent_dim"],
            global_cond_dim=config["world_action_shape"], 
        )
        self.noise_pred_net_eval = ConditionalUnet1D(
            input_dim=config["latent_dim"],
            global_cond_dim=config["world_action_shape"],
        )
        self.nets = nn.ModuleDict({
            'noise_pred_net': self.noise_pred_net
        })
        self.inference_nets = nn.ModuleDict({
            'noise_pred_net': self.noise_pred_net_eval
        })
        
        if(self.config["use_image_cond"]):
            raise NotImplementedError("Image conditioning not implemented")

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
            power=0.75)

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
        if self.config["use_image_cond"] == False:
            return torch.cat([action], dim=-1).flatten(start_dim=1)
        else:
            # TODO: Encode obs through vision encoder if needed
            raise NotImplementedError("Image conditioning not implemented")
        
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
             
            obs_latent = self.vae.get_encoding(obs).unsqueeze(1) # (B, 1, latent_dim)
            cond = self.get_conditioning(obs, action)
                

            if sampler == "ddpm":
                eval_sampler = self.noise_scheduler
                eval_itrs = self.config["num_diffusion_iters"]
            if sampler == "ddim":
                eval_sampler = self.ddim_sampler
                eval_itrs = self.config["num_ddim_iters"]
            
            # init scheduler
            eval_sampler.set_timesteps(eval_itrs)

            # initialize action from obs_latent
            sample = obs_latent
            # sample
            for k in eval_sampler.timesteps:
                # predict noise
                noise_pred = self.inference_nets['noise_pred_net'](
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
            
            return sample
        
    def train_model_step(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor):
        """
        Input: nimage, nagent_pos, naction in the dataset [normalized inputs]
        Train the model for one step
        Returns the loss        
        """
        B = obs.shape[0]
        
        # Get encoded features
        obs_latent = self.vae.get_encoding(obs).unsqueeze(1) # (B, 1, latent_dim)
        next_obs_latent = self.vae.get_encoding(next_obs).unsqueeze(1) # (B, 1, latent_dim)
        cond = self.get_conditioning(obs, action) # (B, action_dim)
        
        # Initialize noisy latent with the observed features
        noise = obs_latent

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=self.device
        ).long()

        # Forward diffusion step: get noise added to the latent at each timestep
        noise_next_obs_latent = self.noise_scheduler.add_noise(next_obs_latent, noise, timesteps) 

        # Predict back the noise
        noise_pred = self.nets['noise_pred_net'](next_obs_latent, timesteps, global_cond=cond)

        # L2 loss
        loss = F.mse_loss(noise_pred, noise_next_obs_latent)

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
    
    def eval_model(self, obs, action, next_obs, sampler="ddim"):
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
            
            # Compute the reconstruction loss
            next_obs_only = next_obs[:, self.config["n_channel"]*(self.config["history_length"]-1):] # last observation
            recon_loss = F.mse_loss(decoded_model_pred, next_obs_only)
            
            losses = {
                "latent_loss": latent_loss.item(),
                "recon_loss": recon_loss.item()
            }
            return losses

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