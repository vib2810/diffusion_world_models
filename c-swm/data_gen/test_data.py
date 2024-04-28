import numpy as np
import matplotlib.pyplot as plt
import h5py
root = 'data/'

file = 'balls_train.h5'

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
# Now list_of_dicts contains the list of dictionaries
for i, dict in enumerate(list_of_dicts):
    print(dict.keys())
    print(dict['obs'].shape)
    print(dict['next_obs'].shape)
    print(dict['action'].shape)
    # images = []
    images = dict['obs'].transpose(0, 2, 3, 1)
    img = np.hstack(images)
    plt.imsave(f'temp/{i}.png', img)

# data = np.load(root + file)
# print(data['train_x'].shape)
# print(data['valid_x'].shape)
# print(data['test_x'].shape)

# train_x = data['train_x']


# for i in range(len(train_x)):
#     print(train_x[i].shape)
#     img = train_x[i][0]
#     plt.imsave('temp.png', img)


# np.savez_compressed(dest,
#                         train_x=sequences[:train_set_size],
#                         valid_x=sequences[
#                                 train_set_size:train_set_size + valid_set_size],
#                         test_x=sequences[train_set_size + valid_set_size:])