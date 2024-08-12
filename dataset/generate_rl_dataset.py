import sys

import numpy as np
from dataset import *

import argparse

sys.path.append('.')
from domains.gridworld import *
from generators.obstacle_gen import *
sys.path.remove('.')

from generators.sparse_map import SparseMap

def gen_kentsommer(dom_size= (8, 8), n_domains=5000, max_obs=50, max_obs_size=2):
    save_path = "dataset/rl/gridworld_{0}x{1}".format(dom_size[0], dom_size[1])

    dom = 0
    images = []
    goals = []

    while dom <= n_domains:
        # generate a random domain
        goal = [np.random.randint(dom_size[0]), np.random.randint(dom_size[1])]
        obs = obstacles([dom_size[0], dom_size[1]], goal, max_obs_size)
        n_obs = obs.add_n_rand_obs(max_obs)
        border_res = obs.add_border()
        if n_obs == 0 or not border_res:
            continue
        im = obs.get_final() # 0 is obstacle 1 is free

        images.append(im)
        goals.append(goal)
        # G = GridWorld(im, goal[0], goal[1])
        # data.append((G)) # G includes image, goal, start, and helper functions.
        # i could even include the obstacle gen in G i guess but its a lot.
        # guarenteed path from start to goal
        # actually don't need to return goal since it's in the image?
        dom += 1
    # images = np.array(images)
    # goals = np.array(goals)
    # np.savez_compressed(save_path, images, goals)
    return images, goals

def gen_koosha(num_envs = 64, map_side_len = 16, obstacle_percent = 20, scale = 2):
    save_path = "dataset/rl/sparsemap_{0}x{1}".format(map_side_len, map_side_len)
    all_envs = SparseMap.genMaps(num_envs, map_side_len, obstacle_percent, scale)
    images = [env.grid for env in all_envs]
    goals = [(env.goal_r, env.goal_c) for env in all_envs]
    # images = np.array(images)
    # goals = np.array(goals)
    # np.savez_compressed(save_path, images, goals)
    return images, goals