# Author: Vibhakar Mohta vib2810@gmail.com
#!/usr/bin/env python3

import time
import os
import numpy as np
import sys
sys.path.append('..')
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm.auto import tqdm
import torch
from diffusion_model import DiffusionTrainer
import torch.utils.data as data
from dataset import StateTransitionsDataset
from model_utils import seed_everything
from collections import defaultdict
import torchvision
import cv2

class ModelEval:
    def __init__(self, config):
        self.config = config
        self.experiment_name_timed = config["experiment_name"]
        self.logdir = 'train_logs/'+ self.experiment_name_timed

        root = f'{config["dataset_root"]}/{config["environment"]}'
        eval_set = StateTransitionsDataset(f'{root}_eval_100ep.h5', mode="eval")
        self.eval_set = eval_set
        self.eval_dataloader = data.DataLoader(eval_set, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
        
        num_batches = len(self.eval_dataloader)
        config["num_batches"] = num_batches
        print("TRAIN BATCHES: ", num_batches)
        print("VAL BATCHES: ", len(self.eval_dataloader))
        # Initialize model
        self.model = DiffusionTrainer(
            config=config,
        )
        self.model.load_model_weights(f'../{self.logdir}/model.pt')
        self.best_eval_loss_recon = 1e10
        self.device = config["device"]
        self.global_step = 0

    def generate_plots(self, start_idx):
        
        obs, action, next_obs = self.eval_set[start_idx]
        # next_obs = next_obs.unsqueeze(0).to(self.device)
        
        # random_action = torch.randint(0, 20, (1,)).to(self.device).unsqueeze(0)
        obs = obs.unsqueeze(0).to(self.device)
        
        random_action = torch.tensor([0]).to(self.device).unsqueeze(0)
        next_obs = next_obs.unsqueeze(0).to(self.device)
        outputs = self.model.eval_model(0, obs, random_action, next_obs, 'ddpm')
        current_latent = outputs['next_obs_latent']
        final_obs = [obs]
        gt_obs = [obs]
        for i in range(1 + start_idx, 10 + start_idx):
            # random_action = torch.randint(0, 20, (1,)).to(self.device).unsqueeze(0)
            _, action, next_obs = self.eval_set[i]
            random_action = torch.tensor([action.item()]).to(self.device).unsqueeze(0)
            next_latent = self.model.get_next_latent(current_latent, random_action, 'ddpm')
            obs = self.model.vae.decoder(next_latent.squeeze(1))
            current_latent = next_latent
            final_obs.append(obs)
            gt_obs.append(next_obs.unsqueeze(0).to(self.device))

        final_obs += gt_obs
        final_obs = torch.stack(final_obs)
        final_obs = final_obs.squeeze(1)

        # write to image
        grid = torchvision.utils.make_grid(
            final_obs, 
            nrow=len(final_obs)//2, normalize=True, 
            pad_value=1, padding=3)
        
        # grid = grid.permute(1, 2, 0)
        # grid = grid.cpu().detach().numpy()
        torchvision.utils.save_image(grid, f'images/final_obs_{start_idx}.png')



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', type=str, default="shapes")
    # parser.add_argument('--expt_name', type=str, required=True)
    parser.add_argument('--dataset_root', type=str, default='/home/punygod_admin/pgm/pgm_project/c-swm/data')
    # parser.add_argument('--history_length', type=int, default=3)
    args = parser.parse_args()

    seed_everything(0)
    
    config = {
        # Dataset
        'dataset_root': args.dataset_root,
        'environment': args.environment,
        
        # Training params
        'lr': 1e-4,
        'num_epochs': 800,
        'batch_size': 512,
        'num_workers': 8,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'eval_sampler': 'ddpm', # 'ddpm' or 'ddim'
        
        # Model params
        'history_length': 1, # Number of frames to stack as input to encoder
        'num_diffusion_iters': 100,
        'num_ddim_iters': 10, # for DDIM sampling
        'vae_pretrained_ckeckpoint': '/home/punygod_admin/pgm/pgm_project/diffusion_world_model/lightning_logs/version_7/checkpoints/epoch96.ckpt',
        'pred_residual': False, # If True, diffuse to next_obs - obs, else diffuse to next_obs
        'latent_dim': 16, # Dimension of the latent space
        'n_channel': 3, # Number of channels in the world image
        'world_image_shape': (64, 64), # Shape of the world image
        'world_action_classes': 20, # size of the action space
        "environment" : args.environment
    }

    # python3 eval_diffusion.py --expt_name big_model_ddpm_ema 
    # config["experiment_name"] = 'diffusion_model' + \
    #                                 '_env_' + config['environment'] + \
    #                                 '_history_' + str(config['history_length']) + \
    #                                 '_latent_' + str(config['latent_dim']) + \
    #                                 '_lr_' + str(config['lr']) +\
    #                                 '_pred_residual_' + str(config['pred_residual']) +\
    #                                 args.expt_name                            
    config['experiment_name'] = 'diffusion_model_env_shapes_history_1_latent_16_lr_0.0001_pred_residual_Falsebig_model_ddpm_ema'
    model_trainer = ModelEval(config)
    for i in range(0, 300, 50):
        model_trainer.generate_plots(i)
