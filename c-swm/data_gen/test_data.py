import numpy as np
import matplotlib.pyplot as plt
import h5py
root = 'data/'

file = 'shapes_train_hist_1.h5'

with h5py.File(root + file, 'r') as hf:
    # Initialize an empty list to store dictionaries
    list_of_dicts = []

    # Iterate through the dataset
    for key in hf:
        dataset = hf[key]
        dic = {}
        for item in dataset:
            # Convert bytes to string and then to dictionary
            dic[item] = dataset[item][()]
        list_of_dicts.append(dic)
import os
os.makedirs('temp', exist_ok=True)
print("length of list_of_dicts:", len(list_of_dicts))
# Now list_of_dicts contains the list of dictionaries
for i, dict in enumerate(list_of_dicts):
    print(dict.keys())
    print(dict['obs'].shape)
    print(dict['next_obs'].shape)
    print(dict['action'].shape)
    # images = []
    images = dict['obs'].transpose(0, 2, 3, 1)
    images *= 255
    images = images.astype(np.uint8)
    for j in range(images.shape[0]):
        if len(np.unique(images[j], axis = 0)) == 5:
            # print(len(np.unique(images[j], axis = 0)))
            # print(np.unique(images[j], axis = 0))
            plt.imsave(f'temp/{i}_{j}.png', images[j])
    img = np.hstack(images)
    plt.imsave(f'temp/{i}.png', img)

