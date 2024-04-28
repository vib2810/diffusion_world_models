"""Simple random agent.

Running this script directly executes the random agent in environment and stores
experience in a replay buffer.
"""

# Get env directory
import sys
from pathlib import Path
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

import argparse

# noinspection PyUnresolvedReferences
import envs

import utils

import gym
from gym import logger

import numpy as np
from PIL import Image
from collections import deque



class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        del observation, reward, done
        return self.action_space.sample()


def crop_normalize(img, crop_ratio = None, img_size = (64, 64)):
    if crop_ratio is not None:
        img = img[crop_ratio[0]:crop_ratio[1]]
    print(img.shape)
    img = Image.fromarray(img).resize(img_size, Image.ANTIALIAS)
    return np.transpose(np.array(img), (2, 0, 1)) / 255



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', type=str, default='ShapesTrain-v0',
                        help='Select the environment to run.')
    parser.add_argument('--fname', type=str, default='data/shapes_train.h5',
                        help='Save path for replay buffer.')
    parser.add_argument('--num_episodes', type=int, default=1000,
                        help='Total number of episodes to simulate.')
    parser.add_argument('--atari', action='store_true', default=False,
                        help='Run atari mode (stack multiple frames).')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed.')
    parser.add_argument('--history_length', type=int, default=3)
    
    args = parser.parse_args()

    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)

    np.random.seed(args.seed)
    env.action_space.seed(args.seed)
    env.seed(args.seed)

    agent = RandomAgent(env.action_space)

    episode_count = args.num_episodes
    reward = 0
    done = False

    crop = None
    warmstart = None
    if args.env_id == 'PongDeterministic-v4':
        crop = (35, 190)
        warmstart = 58
    elif args.env_id == 'SpaceInvadersDeterministic-v4':
        crop = (30, 200)
        warmstart = 50
    else:
        crop = None

    if args.atari:
        env._max_episode_steps = warmstart + 11

    replay_buffer = []

    img_size = (64, 64)
    history_length = args.history_length

    for i in range(episode_count):

        replay_buffer.append({
            'obs': [],
            'action': [],
            'next_obs': [],
        })

        ob = env.reset()

        if args.atari:
            # Burn-in steps
            for _ in range(warmstart):
                action = agent.act(ob, reward, done)
                ob, _, _, _ = env.step(action)
            prev_ob = crop_normalize(ob, crop, img_size)
            ob, _, _, _ = env.step(0)
            ob = crop_normalize(ob, crop, img_size)

        obs_history = deque(maxlen=history_length)
        while True:
            # if args.atari:
            #     ob = crop_normalize(ob, crop, img_size)
            # obs_history.append(ob[1])
            # if len(obs_history) < history_length:
            #     continue
            # obs_concat = np.stack(obs_history, axis=0)
            replay_buffer[i]['obs'].append(ob[1].copy())
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if args.atari:
                ob = crop_normalize(ob, crop, img_size)


            replay_buffer[i]['action'].append(action)
            replay_buffer[i]['next_obs'].append(ob[1].copy())

            if done:
                break
        # print(len(replay_buffer[i]['obs']))
        # replay_buffer[i]['obs'] = np.stack(replay_buffer[i]['obs'])
        # replay_buffer[i]['action'] = np.array(replay_buffer[i]['action'])
        # replay_buffer[i]['next_obs'] = np.stack(replay_buffer[i]['next_obs'])

        # print(replay_buffer[i]['obs'].shape)
        # print(replay_buffer[i]['action'].shape)
        # print(replay_buffer[i]['next_obs'].shape)

        if i % 10 == 0:
            print("iter "+str(i))
    print("writing file")
    env.close()
    fname = args.fname[:-3]
    fname += '_hist_' + str(history_length) + '.h5'
    # Save replay buffer to disk.
    utils.save_list_dict_h5py(replay_buffer, fname)

