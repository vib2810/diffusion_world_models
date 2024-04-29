
import torch.utils.data as data
import h5py
import numpy as np
from vae import VAE
import pytorch_lightning as pl
import torch
from dataset import StateTransitionsDataset

def train(args):
    
    config = {
        'lr': 1e-3,
        'n_epochs': 200,
        'batch_size': 32,
        'n_channel': 3,
        'latent_dim': 16,
    }
    config.update(vars(args)) # add argparse arguments
    root = f'{config["dataset_root"]}/{config["environment"]}'
    train_set = StateTransitionsDataset(f'{root}_train.h5')
    val_set = StateTransitionsDataset(f'{root}_eval.h5')
    model = VAE(
        config = config
    )

    train_loader = data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    trainer = pl.Trainer(
        max_epochs=config['n_epochs'], 
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
    )
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--history_length', type=int, default=1)
    parser.add_argument('--environment', type=str, required=True)
    parser.add_argument('--dataset_root', type=str, default='/home/punygod_admin/pgm/pgm_project/c-swm/data')
    args = parser.parse_args()

    train(args)