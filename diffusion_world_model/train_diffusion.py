# Author: Abhinav Gupta ag6@andrew.cmu.edu
#!/usr/bin/env python3
import sys
from model_trainer import ModelTrainer   
from model_utils import seed_everything
import argparse
import torch
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    print("\n\n")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', type=str, required=True)
    parser.add_argument('--expt_name', type=str, required=True)
    parser.add_argument('--dataset_root', type=str, default='/home/punygod_admin/pgm/pgm_project/c-swm/data')
    parser.add_argument('--history_length', type=int, default=3)
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
    }

    config["experiment_name"] = 'diffusion_model' + \
                                    '_env_' + config['environment'] + \
                                    '_history_' + str(config['history_length']) + \
                                    '_latent_' + str(config['latent_dim']) + \
                                    '_lr_' + str(config['lr']) +\
                                    '_pred_residual_' + str(config['pred_residual']) +\
                                    args.expt_name                            
    model_trainer = ModelTrainer(config)
    model_trainer.train_model()
