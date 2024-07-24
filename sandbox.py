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

config = type('config', (), {'datafile': 'dataset/gridworld_8x8.npz',
                             'imsize': 8,
                             'lr': 0.005, 
                             'epochs': 30,
                             'k': 10,
                             'l_i': 2,
                             'l_h': 150,
                             'l_q': 10,
                             'batch_size': 128})

# predeterminate gridworld yay
start = [2, 5] # [2, 5] is the further start, [4, 5] is the closer start
goal = [5, 2]
img = np.array([[0 for i in range(8)],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 0], 
        [0, 1, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 1, 1, 0],
        [0 for i in range(8)]]) 

# is there a speed difference with list comprehension vs manually init
G = GridWorld(img, goal[0], goal[1])
# simulate the agent moving in the environment for a fixed number of steps
agent = Agent(config)
episodes = 50 # how many training sequences do we wanna generate? basically how many times
                # we wanna complete the maze before being confident we learned the reward func
max_steps = 500 # steps we wanna try before we give up on finding goal
total_steps = 0

# TRAIN
# restart search for n_episodes
# should start at random state, im just setting the start for testing?
# current_state = np.random.randint(0, G.G.shape[0])

for ep in range(episodes): # ep = restart
    current_state = np.int64(np.random.randint(G.G.shape[0])) # G.map_ind_to_state(start[0], start[1])
    done = False
    agent.learn_world(G) # learn inital q func & set up representation of world (so the agent can move?)
    # re do this for each episode/after training so the agent uses the updated
    # q func/rew func to make decisions?
    for step in range(max_steps): # step = 1 action
        total_steps = total_steps + 1
        action = agent.compute_action(G.get_coords(current_state))
        next_state = G.sample_next_state(current_state, action)
        # G.state_to_loc(next_state) #this prints lol
        reward = G.R[current_state][action] # is it bad to directly access... should i make a getter?
        if next_state == G.map_ind_to_state(goal[0], goal[1]):
            done = True
        # agent.store_episode(current_state, action, reward, next_state, done)
        agent.memory_buffer.append({'current_state': G.get_coords(current_state),
                                            'action': action,
                                            'reward': reward,
                                            'next_state': G.get_coords(next_state),
                                            'done': done})
        if done: 
            agent.update_exploration_prob() # add break if done so the agent stops moving
            break
        # then restart cycle/search from begi
        current_state = next_state
    # if we have collected enough training data - train the model
    if total_steps >= config.batch_size: 
        print('training')
        agent.train(config.batch_size, 
                    criterion=nn.CrossEntropyLoss(), 
                    optimizer=optim.RMSprop(agent.model.parameters(), 
                            lr=config.lr, eps=1e-6)) # train model (calculate loss btwn q & target)
    # ^ this is how the example does it, but that means we would start training once
    # a minimum of samples is met, but we would possibly retrain on same data
    # if a new episode doesn't require "batch_size" samples ?
    # so should we reset total_steps each time we train?


# TEST
# ughhh not rlly looking for accuracy i just wanna see how long it takes 
# for the agent to reach the goal i guess

current_state = G.map_ind_to_state(start[0], start[1]) # wanna test w/ preset start
agent.learn_world(G) # creates new q mapping based on "best"/learned reward func
done = False
steps = 0
agent.exploration_prob = 0 # follow policy explicitly now.
while not done:
    action = agent.compute_action(G.get_coords(current_state))
    next_state = G.sample_next_state(current_state, action)
    if next_state == G.map_ind_to_state(goal[0], goal[1]):
        done = True
    current_state = next_state
    print(next_state)
    steps += 1

print('done!')
print(steps)
