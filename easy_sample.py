import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from dataset.dataset import *
from utility.utils import *
from model import *
from sandbox_utils import *

# np.random.seed(9)

config = type('config', (), {'datafile': 'gridworld_8x8.npz',
                             'imsize': 8,
                             'lr': 0.005, 
                             'epochs': 30,
                             'k': 15,
                             'l_i': 2,
                             'l_h': 150,
                             'l_q': 10,
                             'batch_size': 128})


G = generate_gridworld(13, (config.imsize, config.imsize))
agent = Agent(config)

agent, q_target = train_loop(config, G, agent) # more parameters within func like n_episodes, max_steps
pred_actions, target_actions = get_policy(agent, q_target)
print(G.image)

# TEST (on the same gridworld)

agent.learn_world(G) # creates new q mapping based on "best"/learned reward func
# get functional start and optimal path from djikstra
opt_traj, start_state = generate_path(G)
# get predicted path from agent
pred_traj, done_pred = get_pred_path(start_state, G, agent)
# get optimal path from target q values
targ_traj, done_targ = get_target_path(start_state, G, agent, target_actions)
# test if my target is even good...
if targ_traj == opt_traj:
    print('good target?')

# pred = red, targ = blue, dj = green
visualize(G, start = G.get_coords(start_state), goal = [G.target_x, G.target_y], pred_traj = np.array(pred_traj), targ_traj = np.array(targ_traj), opt_traj = np.array(opt_traj))
