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

np.random.seed(9)

config = type('config', (), {'datafile': 'none',
                             'imsize': 8,
                             'lr': 0.005, 
                             'epochs': 30,
                             'k': 10,
                             'l_i': 2,
                             'l_h': 150,
                             'l_q': 10,
                             'batch_size': 128})

# predeterminate gridworld yay
config.imsize = 8
start = [2, 2]
goal = [5, 5]
img = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0], 
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]]) 

# even easier
# config.imsize = 6
# start = [1, 1]
# goal = [4, 4]
# img = np.array ([[0, 0, 0, 0, 0, 0],
#                  [0, 1, 1, 1, 0, 0],
#                  [0, 0, 1, 0, 1, 0],
#                  [0, 0, 0, 0, 1, 0],
#                  [0, 0, 0, 0, 1, 0],
#                  [0, 0, 0, 0, 0, 0]])

G = GridWorld(img, goal[0], goal[1])
agent = Agent(config)
episodes = 200 
max_steps = 50 # steps we wanna try before we give up on finding goal (computational bound)
total_steps = 0

# ACTUAL TRAIN W/ NN
q_target = torch.zeros((config.imsize, config.imsize, 8))
agent.gamma = 0.75

for ep in range(episodes):
    current_state = np.int64(np.random.randint(G.G.shape[0]))
    done = False
    agent.learn_world(G)
    for step in range(max_steps):
        total_steps = total_steps + 1
        action = agent.compute_action(G.get_coords(current_state))
        next_state = G.sample_next_state(current_state, action)
        reward = G.R[current_state][action]
        state_x, state_y = G.get_coords(current_state)
        state_x_, state_y_ = G.get_coords(next_state)
        q_target[state_x][state_y][action] = reward + agent.gamma * max(q_target[state_x_][state_y_])
        if next_state == G.map_ind_to_state(goal[0], goal[1]): 
            done = True
        agent.store_episode((state_x, state_y), action, reward, (state_x_, state_y_), done)
        if done == True:
            agent.update_exploration_prob(decay=0.001)
            break
        current_state = next_state
    if total_steps >= config.batch_size: # should we still train if goal was never reached ? is that useful.
        print('training...')
        agent.train(config.batch_size, 
                    criterion=nn.MSELoss(),  # used to be Cross Entropy, but I think that works better for multiclass/not good for predicing specific values. 
                    optimizer=optim.RMSprop(agent.model.parameters(), 
                            lr=config.lr, eps=1e-6), q_target=q_target)
        total_steps = 0 # reset total steps so it can rebuild memory buffer??? im not sure.
        agent.memory_buffer = [] # and reset memory buffer ? not sure. 
        

# TEST
# ughhh not rlly looking for accuracy i just wanna see how long it takes 
# for the agent to reach the goal i guess

# print policy 
q_values = agent.q_values
pred_q = [agent.model.fc(q_values[0, :, i, j]).detach().numpy() for i in range(config.imsize) for j in range(config.imsize)]
# q_final = [agent.model.fc(q_values[0, :, j, i]).detach().numpy() for i in range(config.imsize) for j in range(config.imsize)]
pred_actions = np.argmax(pred_q, axis=1)
# format like the grid
print(pred_actions)
pred_actions = np.array([pred_actions[i:i+config.imsize] for i in range(0, len(pred_actions), config.imsize)])
target_actions = np.argmax(q_target, axis=2)
print(pred_actions)
print(target_actions)
# test trajectory
current_state = G.map_ind_to_state(start[0], start[1]) # wanna test w/ preset start
agent.learn_world(G) # creates new q mapping based on "best"/learned reward func
done = False
steps = 0
agent.exploration_prob = 0 # follow policy explicitly now.
while not done and steps < 10:
    print(current_state)
    action = agent.compute_action(G.get_coords(current_state))
    # action = np.argmax(q_values[current_state])
    print('action', action)
    next_state = G.sample_next_state(current_state, action)
    if next_state == G.map_ind_to_state(goal[0], goal[1]):
        done = True
        print('solved!')
    current_state = next_state
    steps += 1


print(steps)
