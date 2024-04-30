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

        # Initialize the writer
        # curr_time= time.strftime("%d-%m-%Y_%H-%M-%S")
        # self.experiment_name_timed = config["experiment_name"] + "_" + curr_time
        self.experiment_name_timed = config["experiment_name"]
        self.logdir = 'train_logs/'+ self.experiment_name_timed
        if not(os.path.exists(self.logdir)):
            os.makedirs(self.logdir)
        print("Logging to: ", self.logdir)
        self.writer = SummaryWriter(self.logdir)
        
        # Initialize the dataset
        root = f'{config["dataset_root"]}/{config["environment"]}'
        train_set = StateTransitionsDataset(f'{root}_train.h5')
        val_set = StateTransitionsDataset(f'{root}_eval.h5', mode="eval")
        self.train_dataloader = data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
        self.eval_dataloader = data.DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
        
        num_batches = len(self.train_dataloader)
        config["num_batches"] = num_batches
        print("TRAIN BATCHES: ", num_batches)
        print("VAL BATCHES: ", len(self.eval_dataloader))

        # Initialize model
        self.model = DiffusionTrainer(
            config=config,
        )
        
        self.best_eval_loss_recon = 1e10
        self.device = config["device"]
        self.store_model_normalization()
    
    def store_model_normalization(self):
        print("Computing Normalization Parameters")
        for nbatch in self.train_dataloader:
            obs, action, next_obs = nbatch
            obs = obs.float().to(self.device)
            next_obs = next_obs.float().to(self.device)
            self.model.get_normalization_params_step(obs, next_obs)
        
        for nbatch in self.eval_dataloader:
            obs, action, next_obs = nbatch
            obs = obs.float().to(self.device)
            next_obs = next_obs.float().to(self.device)
            self.model.get_normalization_params_step(obs, next_obs)
        
        self.model.print_normalization_params()

    def train_model(self):
        self.global_step = 0
        for epoch_idx in range(self.config["num_epochs"]):
            print("-----Epoch {}-----".format(epoch_idx))
            
            # evaluate model
            eval_losses, stacked_samples = self.evaluate_model()
            for k, v in eval_losses.items():
                self.writer.add_scalar(f'Loss/{k}', v, self.global_step)
            # log stacked_samples as an image
            if stacked_samples is not None:
                self.writer.add_image('recon_samples', stacked_samples, self.global_step)
                
            print("Eval losses: Latent: {}, Recon: {}".format(eval_losses["latent_loss"], eval_losses["recon_loss"]))
            if eval_losses["recon_loss"] < self.best_eval_loss_recon:
                self.best_model_epoch = epoch_idx
                self.best_eval_loss_recon = eval_losses["recon_loss"]
                self.save_model()
                
            # Batch loop
            for nbatch in self.train_dataloader:
                # Extract data
                obs, action, next_obs = nbatch
                
                obs = obs.float().to(self.device)
                action = action.to(self.device)
                next_obs = next_obs.float().to(self.device)
                
                loss_cpu = self.model.train_model_step(obs, action, next_obs, self.global_step)
                
                # log to tensorboard
                self.writer.add_scalar('Loss/train', loss_cpu, self.global_step)
                self.global_step += 1
                
                if(not self.global_step%20):
                    print("Epoch: {}, Step: {}, Loss: {}".format(epoch_idx, self.global_step, loss_cpu))
            
            # evaluate model on test data
            self.model.run_after_epoch()
                
    def save_model(self, step=None):
        save_dict = {}
        save_dict["model_weights"] = self.model.state_dict()
        save_dict["best_model_epoch"] = self.best_model_epoch
        save_dict["norm_max"] = self.model.norm_max
        save_dict["norm_min"] = self.model.norm_min
        
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
        stacked_samples = None
        for idx, nbatch in enumerate(self.eval_dataloader):
                # Extract data
            obs, action, next_obs = nbatch # shapes(B,3,64,64), (B), (B,3,64,64)
            
            obs = obs.float().to(self.device)
            action = action.to(self.device)
            next_obs = next_obs.float().to(self.device)
                
            B = obs.shape[0]
            save = True if idx == 1 else False
            output_dic = self.model.eval_model(self.global_step, obs, action, next_obs, save=save, sampler=self.config['eval_sampler'])
            stacked_i = output_dic['stacked']
            losses = output_dic['losses']
            if save:
                stacked_samples = stacked_i
            # multiply by batch size
            for k, v in losses.items():
                total_loss[k] += v*B
        
        # Normalize the losses
        normalized_losses = {k: v/len(self.eval_dataloader.dataset) for k, v in total_loss.items()}
        return normalized_losses, stacked_samples