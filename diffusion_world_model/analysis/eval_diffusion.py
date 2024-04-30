# Author: Vibhakar Mohta vib2810@gmail.com
#!/usr/bin/env python3

import time
import os
import utils
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm.auto import tqdm
import torch
from diffusion_model import DiffusionTrainer
import torch.utils.data as data
from dataset import StateTransitionsDataset
from model_utils import seed_everything
from collections import defaultdict

class ModelEval:
    def __init__(self, config):
        self.config = config
        self.experiment_name_timed = config["experiment_name"]
        self.logdir = 'train_logs/'+ self.experiment_name_timed

        root = f'{config["dataset_root"]}/{config["environment"]}'
        eval_set = StateTransitionsDataset(f'{root}_eval_100ep.h5', mode="eval")
        self.eval_dataloader = data.DataLoader(eval_set, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
        
        num_batches = len(self.eval_dataloader)
        config["num_batches"] = num_batches
        print("TRAIN BATCHES: ", num_batches)
        print("VAL BATCHES: ", len(self.eval_dataloader))
        # Initialize model
        self.model = DiffusionTrainer(
            config=config,
        )
        self.model.load_model_weights(f'{self.logdir}/epoch_770.pt')
        self.best_eval_loss_recon = 1e10
        self.device = config["device"]
        self.global_step = 0

    def evaluate_model(self):
        """
        Evaluates a given model on a given dataset
        Saves the model if the test loss is the best so far
        """
        # Evaluate the model by computing the MSE on test data
        topk = [1]
        pred_states = []
        next_states = []
        num_samples = 0
        hits_at = defaultdict(int)
        rr_sum = 0
        mse_recon_sum = 0
        mse_latent_sum = 0
        for idx, nbatch in enumerate(self.eval_dataloader):
                # Extract data
            obs, action, next_obs = nbatch # shapes(B,3,64,64), (B), (B,3,64,64)
            
            obs = obs.float().to(self.device)
            action = action.to(self.device)
            next_obs = next_obs.float().to(self.device)
                
            B = obs.shape[0]
            save = True if idx == 1 else False
            output_dic = self.model.eval_model(self.global_step, obs, action, next_obs, save=save, sampler=self.config['eval_sampler'])
            
            pred_states.append(output_dic["model_pred"].clone())
            next_states.append(output_dic["next_obs_latent"].clone())
            mse_latent_sum += output_dic["losses"]['latent_loss']
            mse_recon_sum += output_dic["losses"]['recon_loss']

        pred_state_cat = torch.cat(pred_states, dim=0)
        next_state_cat = torch.cat(next_states, dim=0)

        full_size = pred_state_cat.size(0)

        # Flatten object/feature dimensions
        next_state_flat = next_state_cat.view(full_size, -1)
        pred_state_flat = pred_state_cat.view(full_size, -1)

        dist_matrix = utils.pairwise_distance_matrix(
            next_state_flat, pred_state_flat)
        dist_matrix_diag = torch.diag(dist_matrix).unsqueeze(-1)
        dist_matrix_augmented = torch.cat(
            [dist_matrix_diag, dist_matrix], dim=1)

        # Workaround to get a stable sort in numpy.
        dist_np = dist_matrix_augmented.detach().cpu().numpy()
        indices = []
        for row in dist_np:
            keys = (np.arange(len(row)), row)
            indices.append(np.lexsort(keys))
        indices = np.stack(indices, axis=0)
        indices = torch.from_numpy(indices).long()

        labels = torch.zeros(
            indices.size(0), device=indices.device,
            dtype=torch.int64).unsqueeze(-1)

        num_samples += full_size
        print('Size of current topk evaluation batch: {}'.format(
            full_size))

        for k in topk:
            match = indices[:, :k] == labels
            num_matches = match.sum()
            hits_at[k] += num_matches.item()

        match = indices == labels
        _, ranks = match.max(1)

        reciprocal_ranks = torch.reciprocal(ranks.double() + 1)
        rr_sum += reciprocal_ranks.sum()

        pred_states = []
        next_states = []
        for k in topk:
            print('Hits @ {}: {}'.format(k, hits_at[k] / float(num_samples)))

        print('MRR: {}'.format(rr_sum / float(num_samples)))
        print("MSE_LATENT: ", mse_latent_sum/num_samples)
        print("MSE_RECON: ", mse_recon_sum/num_samples)
    

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
        'pred_residual': True, # If True, diffuse to next_obs - obs, else diffuse to next_obs
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
    model_trainer.evaluate_model()
