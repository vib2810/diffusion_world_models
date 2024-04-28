# Author: Vibhakar Mohta vib2810@gmail.com
#!/usr/bin/env python3

import time
import os
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm.auto import tqdm
import torch
from diffusion_model import DiffusionTrainer
import torch.utils.data as data
from dataset import StateTransitionsDataset

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.is_state_based = config["is_state_based"]

        # Initialize the writer
        curr_time= time.strftime("%d-%m-%Y_%H-%M-%S")
        self.experiment_name_timed = config["experiment_name"] + "_" + curr_time
        self.logdir = 'train_logs/'+ self.experiment_name_timed
        if not(os.path.exists(self.logdir)):
            os.makedirs(self.logdir)
        print("Logging to: ", self.logdir)
        self.writer = SummaryWriter(self.logdir)
        
        # Initialize the dataset
        root = f'{config["dataset_root"]}/{config["environment"]}'
        train_set = StateTransitionsDataset(f'{root}_train_hist_{config["history_length"]}.h5')
        val_set = StateTransitionsDataset(f'{root}_eval_hist_{config["history_length"]}.h5')
        self.train_dataloader = data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=4)
        self.eval_dataloader = data.DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, num_workers=4)
        
        num_batches = len(self.train_dataloader)
        config["num_batches"] = num_batches

        # Initialize model
        self.model = DiffusionTrainer(
            config=config,
        )
        
        self.best_eval_loss_recon = 1e10
        self.device = config["device"]

    def train_model(self):
        global_step = 0
        for epoch_idx in range(self.config["num_epochs"]):
            print("-----Epoch {}-----".format(epoch_idx))
            
            # evaluate model
            eval_losses = self.evaluate_model()
            for k, v in eval_losses.items():
                self.writer.add_scalar(f'Loss/{k}', v, global_step)
                
            print("Eval losses: Latent: {}, Recon: {}".format(eval_losses["latent_loss"], eval_losses["recon_loss"]))
            if eval_losses["recon_loss"] < self.best_eval_loss_recon:
                self.best_model_epoch = epoch_idx
                self.best_eval_loss_recon = eval_losses["recon_loss"]
                self.save_model()
                
            # Batch loop
            for nbatch in self.train_dataloader:
                # Extract data
                obs, action, next_obs = nbatch
                # save next_obs as an image
                import matplotlib.pyplot as plt
                plt.figure()
                plt.imshow(next_obs[0].permute(1, 2, 0))
                plt.savefig("next_obs.png")
                
                # create next_obs_stacked
                next_obs_stacked = torch.cat([obs[:, self.config["n_channel"]:], next_obs], dim=1) # make stacked next_obs
                next_obs_stacked = next_obs_stacked.detach()
                
                obs = obs.float().to(self.device)
                action = action.to(self.device)
                next_obs_stacked = next_obs_stacked.float().to(self.device)
                
                loss_cpu = self.model.train_model_step(obs, action, next_obs_stacked)
                
                # log to tensorboard
                self.writer.add_scalar('Loss/train', loss_cpu, global_step)
                global_step += 1
                
                if(not global_step%20):
                    print("Epoch: {}, Step: {}, Loss: {}".format(epoch_idx, global_step, loss_cpu))
            
            # evaluate model on test data
            self.model.run_after_epoch()
                
    def save_model(self, step=None):
        save_dict = {}
        save_dict["model_weights"] = self.model.state_dict()
        save_dict["best_model_epoch"] = self.best_model_epoch
        
        # add train params to save_dict
        save_dict.update(self.config)
            
        torch.save(save_dict, self.logdir + "/model.pt")

    def evaluate_model(self):
        """
        Evaluates a given model on a given dataset
        Saves the model if the test loss is the best so far
        """
        # Evaluate the model by computing the MSE on test data
        total_loss = {"latent_loss": 0, "recon_loss": 0}
        for nbatch in self.eval_dataloader:
            obs, action, next_obs = nbatch
            next_obs_stacked = torch.cat([obs[:, self.config["n_channel"]:], next_obs], dim=1) # make stacked next_obs
            next_obs_stacked = next_obs_stacked.detach()
            
            obs = obs.float().to(self.device)
            action = action.to(self.device)
            next_obs_stacked = next_obs_stacked.float().to(self.device)
            
            B = obs.shape[0]

            losses = self.model.eval_model(obs, action, next_obs_stacked)
            # multiply by batch size
            for k, v in losses.items():
                total_loss[k] += v*B
        
        # Normalize the losses
        normalized_losses = {k: v/len(self.eval_dataloader.dataset) for k, v in total_loss.items()}
        return normalized_losses