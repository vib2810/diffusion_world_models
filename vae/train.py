
import torch.utils.data as data
import h5py
import numpy as np
from model import VAE
import pytorch_lightning as pl
def load_list_dict_h5py(fname):
    """Restore list of dictionaries containing numpy arrays from h5py file."""
    array_dict = list()
    with h5py.File(fname, 'r') as hf:
        for i, grp in enumerate(hf.keys()):
            array_dict.append(dict())
            for key in hf[grp].keys():
                array_dict[i][key] = hf[grp][key][:]
    return array_dict


class StateTransitionsDataset(data.Dataset):
    """Create dataset of (o_t, a_t, o_{t+1}) transitions from replay buffer."""

    def __init__(self, hdf5_file):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience
                buffer
        """
        self.experience_buffer = load_list_dict_h5py(hdf5_file)

        # Build table for conversion between linear idx -> episode/step idx
        self.idx2episode = list()
        step = 0
        for ep in range(len(self.experience_buffer)):
            num_steps = len(self.experience_buffer[ep]['action'])
            idx_tuple = [(ep, idx) for idx in range(num_steps)]
            self.idx2episode.extend(idx_tuple)
            step += num_steps

        self.num_steps = step

    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx):
        ep, step = self.idx2episode[idx]
        obs = (self.experience_buffer[ep]['obs'][step]).astype(np.float32) # [3 * n_stack, 64, 64]
        action = self.experience_buffer[ep]['action'][step] # [1] - index of action
        next_obs = (self.experience_buffer[ep]['next_obs'][step]).astype(np.float32) # [3 * n_stack, 64, 64]
        return obs, action, next_obs


def train():
    dataset_root = ''
    environment = ''
    history_length = 3

    train_set = StateTransitionsDataset(f'{dataset_root}/{environment}_train_hist_{history_length}.h5')
    val_set = StateTransitionsDataset(f'{dataset_root}/{environment}_eval_hist_{history_length}.h5')
    model = VAE(n_stack=3)

    train_loader = data.DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = data.DataLoader(val_set, batch_size=32, shuffle=False)

    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    train()