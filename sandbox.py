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


config = type('config', (), {'datafile': 'dataset/rl/gridworld_8x8.npz',
                             'imsize': 8,
                             'lr': 0.005, 
                             'epochs': 30,
                             'k': 15,
                             'l_i': 2,
                             'l_h': 150,
                             'l_q': 10,
                             'batch_size': 128})


# load data
trainset = GridworldData(config.datafile, imsize=config.imsize, train=True)
testset = GridworldData(config.datafile, imsize=config.imsize, train=False)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=False)

# train on random n worlds
agent = Agent(config)
for epoch in range(config.epochs):
    avg_error, avg_loss, num_batches = 0
    for i, data in enumerate(trainloader):
        image, start, goal = data
        if image.size()[0] != config.batch_size:
            continue # Drop those data, if not enough for a batch
        agent, q_target = train_loop(config, image, agent)
        pred_actions, target_actions = get_policy(agent, q_target)
        agent.learn_world(G)
        opt_traj, start_state = generate_path(G)
        if opt_traj == False: 
            continue
        pred_traj, done_pred = get_pred_path(start_state, G, agent)
        target_traj, done_target = get_target_path(start_state, G, agent, target_actions)
        if done_pred == True:
            print('yay')

n_worlds = 50
count_correct = 0

agent = Agent(config)
for n in range(n_worlds):
    G = generate_gridworld(16, (config.imsize, config.imsize))
    agent, q_target = train_loop(config, G, agent) # more parameters within func
    pred_actions, target_actions = get_policy(agent, q_target)
    # test
    agent.learn_world(G)
    opt_traj, start_state = generate_path(G)
    if opt_traj == False:
        continue
    pred_traj, done_pred = get_pred_path(start_state, G, agent)
    target_traj, done_target = get_target_path(start_state, G, agent, target_actions)
    if done_pred == done_target:
        count_correct += 1

print(f'Train Accuracy: {count_correct/n_worlds}')

# test on new world (s)
count_correct = 0
for world in range (20):
    G = generate_gridworld(16, (config.imsize, config.imsize))
    agent.learn_world(G)
    opt_traj, start_state = generate_path(G)
    pred_traj, done_pred = get_pred_path(start_state, G, agent)
    # target_traj, done_target = get_target_path(start_state, G, agent, target_actions)
    if done_pred == True:
        print('yay')
    else: 
        print('nay')
    count_correct += 1 if done_pred == True else 0

print('Test Accuracy:', count_correct/20)