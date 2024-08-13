import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from utility.utils import *
from sandbox_utils import *

# WIP
from generators.sparse_map import SparseMap
from domains.batch_worlds import *

np.random.seed(9)
config = type('config', (), {'datafile': 'dataset/rl/gridworld_8x8.npz',
                             'imsize': 8,
                             'lr': 0.005, 
                             'epochs': 30,
                             'k': 15,
                             'l_i': 2,
                             'l_h': 150,
                             'l_q': 10,
                             'batch_size': 128})

# generate data?? -sparse map not kentsommer gridworld.
num_envs = 16
config.batch_size = 4
map_side_len = 16
obstacle_percent = 20
scale = 2
all_envs = SparseMap.genMaps(num_envs, map_side_len, obstacle_percent, scale)
# SparseMap generates 0 = freespace, 1 = obstacle, which is the opposite encoding of the
# worlds object ... :|
images = np.array([env.grid for env in all_envs])
images = np.logical_not(images)
goals = np.array([(env.goal_r, env.goal_c) for env in all_envs])
trainset = DatasetFromArray(images, goals, train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True)

# more config info
episodes = 100
max_steps = 50
total_steps = 0
decay = 0.99 # for exploration prob

# train on random n worlds
model = VIN(config) # list of agent objects with just a nested list because I feel like array accessing is easier this wayz

# okay this whole thing feels really stupid because i'm using list comprehension to do each computation on each world
# I guess it would be possible to rewrite the GridWorld class to accept a batch of images and goals and then do the computation
# ^ like write it so everything is designed to accept an extra dimension and it would be faster then
# UGh math.


trainloader = trainset[0:4]
images, goals = trainloader[0], trainloader[1]
worlds = Worlds(images, goals[:, 0], goals[:, 1])
exploration_prob = 1.0
current_state = worlds.loc_to_state(worlds.start_x, worlds.start_y)
q_target = torch.zeros((worlds.n_worlds, worlds.n_row, worlds.n_col, 8)) # batch, states, actions
done = np.array([False]*worlds.n_worlds)
q_values = model.map_q(worlds.iv_mixed, config.k)
step = 0
total_steps += 1
if np.random.uniform(0, 1) < exploration_prob:
    action = np.random.choice(8, size=4)

# get next state by transition matrix
trans_probabilities = worlds.P[0, current_state, :, action] # T(s, a, *)
next_state = 


rng = np.random.default_rng()
next_state = rng.choice(range(worlds.n_states), axis=1, )


for epoch in range(config.epochs): # loop over the dataset multiple times
    avg_error, avg_loss, num_batches = 0, 0, 0
    for i, data in enumerate(trainloader): # for each batch in dataset
        image, goal = data # image, goal = trainset.images[0:4], trainset.goals[0:4]
        worlds = Worlds(image, goal[:, 0], goal[:, 1]) # one worlds object per batch
        if worlds.n_worlds != config.batch_size:
            continue # Drop those data, if not enough for a batch
            # ^ before this was worlds.size idk whhy
        exploration_prob = 1.0
        for ep in range(episodes): # do a certain number of episodes (restart from beginning)
            # same as old code, but it's in batches now so there are extra dimensions
            current_state = worlds.loc_to_state(worlds.start_x, worlds.start_y)
            q_target = torch.zeros((worlds.n_worlds, worlds.n_row, worlds.n_col, 8)) # batch, states, actions
            done = np.array([False]*worlds.n_worlds)
            q_values = model.map_q(worlds.iv_mixed, config.k)
            for step in range(max_steps):
                total_steps = total_steps + 1
                # choose action
                if np.random.uniform(0, 1) < exploration_prob:
                    action = np.random.choice(8)
                else:
                    current_x, current_y = np.array([G.get_coords(current_state) for G in worlds]).reshape((worlds.shape[0], 2)) # uhhh dunno if this makes sense haha
                    _, action = model.forward(worlds.iv_mixed, current_x, current_y, config.k)
                    action = np.argmax(action.detach().numpy())
                
        

            
            
            
            
            
            
            
            
            
#################################################


            current_state = G.map_ind_to_state(G.start_x, G.start_y) # generate multiple starts or no? easily implemented by changing start x/y to list of connected component states and just picking random options
            q_target = torch.zeros((G.shape[0], G.n_rows, G.n_cols, 8))
            done = np.array([False]*G.shape[0])
            
            q_values = agent.model.map_q(G.iv_mixed, config.k) #HHH WRONG cuz its an array of agents ?
            for step in range(max_steps):
                total_steps = total_steps + 1
                # choose action
                if np.random.uniform(0, 1) < exploration_prob:
                    action = np.random.choice(8)
                else:
                    current_x, current_y = G.get_coords(current_state)
                    _, action = agent.model.forward(G.iv_mixed, current_x, current_y, config.k)
                    action = np.argmax(action.detach().numpy())
                next_state = G.next_state(current_state, action)
                next_x, next_y = G.get_coords(next_state)
                reward = G.R[current_state, action] # check this works properly with the array
                new_target = q_target.clone()
                new_target[current_x, current_y] = reward + agent.gamma * torch.max(q_target[next_x, next_y])
                q_target = np.where(done == True, q_target[current_x, current_y], new_target[current_x, current_y])
                # surely there is a better way to write this.
                for i in range(agent.size):
                    if done[i] == False:
                        if current_state[i] == G.map_ind_to_state(G.target_x, G.target_y): # tbh i dont need a dict for this i could just use an array
                            agent[i].store_episode((current_x[i], current_y[i]),
                                                action[i],
                                                reward[i],
                                                (next_x[i], next_y[i]),
                                                True)
                        else:
                            agent[i].store_episode((current_x[i], current_y[i]),
                                                action[i],
                                                reward[i],
                                                (next_x[i], next_y[i]),
                                                False)
                    # stop collecting experiences if it's already reached the trajectory I guess
                    # but it's so wasteful to simulate the action if it s laready doneeeee, but makes the code cleaner by computing
                    # across the whole array ... :|
                done = np.where(next_state == G.map_ind_to_state(G.target_x, G.target_y) or done == True, True, False)
                current_state = next_state
                # need agent.q_target somewhere, or just store q_target in the experience array... or something.
                
            exploration_prob = np.exp(-decay)
            # BEGIN TRAINING: after the agent explores a lot and reaches the goal (or not)
            # optional: filter for only the successful trialss?
            
            # the logic is:
            # for each agent in the array, shuffle their experiences >> could use 
            batched_data = []
            for agent_ in agent:
                np.random.permutation(agent_.memory_buffer)
                batched_data.append(agent_.memory_buffer)

            for i in range(batched_data[0].shape[0]): # the len/range is wrong bc theyre all different . what is the cut off idk.
                [agent[i].model.map(lambda x, G: x.map_q(G.iv_mixed, config.k), )]
                


            # then for each agent, get first experience 




            # okay where batching comes in, is when the model is able to use multiple gridworlds to calculate map_qs
            # shuffle exeriences
            




            
                





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