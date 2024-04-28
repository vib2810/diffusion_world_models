
import h5py
import numpy as np
from torchvision import transforms
import torch.utils.data as data
import torch
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

        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

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
        obs = (self.experience_buffer[ep]['obs'][step]).astype(np.float32) # [3, 64, 64]
        obs = torch.tensor(obs)
        obs = self.transform(obs)
        action = self.experience_buffer[ep]['action'][step] # [1] - index of action
        if len(action.shape) == 0:
            action = np.array([action]).astype(np.int64)
            action = torch.tensor(action)
        next_obs = (self.experience_buffer[ep]['next_obs'][step]).astype(np.float32) # [3, 64, 64]
        next_obs = torch.tensor(next_obs)
        next_obs = self.transform(next_obs)
        return obs, action, next_obs
    

if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    os.makedirs('temp', exist_ok=True)
    root = '/home/punygod_admin/pgm/pgm_project/c-swm/data/'
    file = 'shapes_temptest_hist_1.h5'
    train_set = StateTransitionsDataset(root + file)
    for i in range(len(train_set)):
        obs, action, next_obs = train_set[i]
        print(obs.shape, action.shape, next_obs.shape)
        obs -= obs.min()
        obs /= obs.max()
        obs = obs.permute(1, 2, 0).cpu().detach().numpy()
        next_obs -= next_obs.min()
        next_obs /= next_obs.max()
        next_obs = next_obs.permute(1, 2, 0).cpu().detach().numpy()
        stacked = np.vstack([obs, np.ones_like(obs), next_obs])
        plt.imsave(f'temp/{i}.png', stacked)
        if i == 10:
            break