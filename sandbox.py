import numpy as np
import matplotlib.pyplot as plt
import torch
import logging
from datetime import datetime

from generators.sparse_map import  SparseMap
from model import VIN

# generate multiple worlds uh oh
num_envs = 32
map_side_len = 8
obstacle_percent = 20
scale = 2

envs = SparseMap.genMaps(num_envs, map_side_len, obstacle_percent, scale) # 0 = 0 freespace, 1 = obstacle
worlds = [env.genPOMDP() for env in envs] # equivalent to what "gridworld" was, stores all info like T, O, R, etc.
    # i sort of don't like how this is implemented ? only 4 actions, not "in order"
    # indexing is [a, s, s'] instead of [s, a, s']
    # gridworld seems fancier with how it gets the transitions but this also makes more sense maybe...
    # also could remove stuff like "addPath" etc because we don't need expert solvers

    # twoD_map_path is more focused on representing the expert paths and actions that I don't need for my implementation
    # is it worth it to rewrite/reorganize the code so the "world" class is just basic map info/functions, gridworld/pomdp representations, and image view?

action_mapping = ["right", "up", "left", "down", "stay"]
# one world at a time ?
# need to pair world info with trajectory info tho

config = {
    "imsize": map_side_len + 2, 
    "n_act": 5, 
    "lr": 0.005,
    'epochs': 30,
    'l_i': 2,
    'l_h': 150,
    "l_q": 10,
    "k": 20,
}

model = VIN(config)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

for epoch in range(50):
    print('epoch:', epoch)
    trajectories = []
    for world, env in zip(worlds, envs):
        start_state = np.random.randint(len(world.states)) # random start state idx
        goal_state = SparseMap.rcToRoomIndex(env.grid, env.goal_r, env.goal_c)
        n_traj = 10
        max_steps = 5000
        for traj in range(n_traj):
            trajectory_time = datetime.now()
            # should i regenerate the start and goal states each time?
            current_state = start_state
            memories = []
            total_steps = 0
            done = False
            while done == False & total_steps < max_steps:
                step_time = datetime.now()
                ### if u loop this --> line 42 it will be a random walk simulation 
                action = np.random.choice(world.n_actions) # random action idx
                next_state = np.random.choice(range(world.n_states), p=world.T[action, current_state]) # next state based on action and current state
                observation = np.random.choice(range(len(world.observations)), p=world.O[action, next_state]) # um what are these observations/what do they mean...
                reward = world.R[action, current_state, next_state, observation]
                current_state = next_state
                # end at goal
                if current_state == goal_state:
                    done = True
                memories.append({
                    'current_state': current_state,
                    'action': action,
                    'next_state': next_state,
                    'reward': reward,
                    'done': done
                })
                total_steps += 1
            if done == True: # currently only store info if its successful
                trajectories.append({
                    'trajectory': memories,
                    'world': world, # is it potentially bad to do this because storing multiples of world/env? idk but it works for now maybe
                    'env': env
                })
            # store data as 1 complete trajectory full of multiple memories in each entry
            # can shuffle the trajectories and also the memeories within the trajectory ?

    # initialize the model for training after experience collecting is done

    # TRAIN?
    print('training...', len(trajectories))
    np.random.shuffle(trajectories) # shuffle the worlds, (important cuz same world is repeated in multiple trajectories in order.)
    # for each trajectory: 
    for item in trajectories:
        trajectory = item['trajectory']
        world = item['world']
        env = item['env']
        # create input view
        reward_mapping = -1 * np.ones(env.grid.shape) # -1 for freespace
        reward_mapping[env.goal_r, env.goal_c] = 10 # 10 at goal, if regenerating goals for each world then i'd need to redo this for each goal/trajectory.
        grid_view = np.reshape(env.grid, (1, 1, env.grid.shape[0], env.grid.shape[1]))
        reward_view = np.reshape(reward_mapping, (1, 1, env.grid.shape[0], env.grid.shape[1]))
        input_view = torch.Tensor(np.concatenate((grid_view, reward_view), axis=1)) # inlc empty 1 dim
        # would be nice to store this/method for this directly in world object
        np.random.shuffle(trajectory) # shuffle the experiences within a trajectory too ? 
        for experience in trajectory:
            # since i update weights, do I need to reacalcculate all of these ech time omg.
            r, v = model.process_input(input_view)
            for i in range(config['k'] - 1):
                q = model.eval_q(r, v)
                v, _ = torch.max(q, dim=1, keepdim=True)
            q = model.eval_q(r, v) 
            # train/learn for each experience
            optimizer.zero_grad() # when to do this. now or in traj loops?
            state_x, state_y = SparseMap.roomIndexToRc(env.grid, experience['current_state']) # should I directly store states as tuple coords? maybe.
            next_state_x, next_state_y = SparseMap.roomIndexToRc(env.grid, experience['next_state'])
            q_pred, _ = model.get_action(q, state_x, state_y)
            q_target = experience['reward'] # pull experiences from stored actions
            if not experience['done']:
                q_target = q_target + world.discount * np.max(q[0, :, next_state_x, next_state_y].detach().numpy())

            q_pred[0, experience['action']] = q_target # first dim is env idx, bc theres an extra dim for batch size. how to combine ahhhhh
            loss = criterion(model.get_action(q, state_x, state_y)[0], q_pred)
            loss.backward()
            optimizer.step()

# test >>????
# create new testing env (will perform badly if trained on only one env tho duh)
env = SparseMap.genMaps(num_envs, map_side_len, obstacle_percent, scale)[0] # 0 = 0 freespace, 1 = obstacle
world = env.genPOMDP()
start_state = np.random.randint(len(world.states)) # random start state idx
goal_state = SparseMap.rcToRoomIndex(env.grid, env.goal_r, env.goal_c)


# create input view
reward_mapping = -1 * np.ones(env.grid.shape) # -1 for freespace
reward_mapping[env.goal_r, env.goal_c] = 10 # 10 at goal, if regenerating goals for each world then i'd need to redo this for each goal/trajectory.
grid_view = np.reshape(env.grid, (1, 1, env.grid.shape[0], env.grid.shape[1]))
reward_view = np.reshape(reward_mapping, (1, 1, env.grid.shape[0], env.grid.shape[1]))
input_view = torch.Tensor(np.concatenate((grid_view, reward_view), axis=1)) # inlc empty 1 dim

# would be nice to store this/method for this directly in world object
# learn world
r, v = model.process_input(input_view)
for i in range(config['k'] - 1):
    q = model.eval_q(r, v)
    v, _ = torch.max(q, dim=1, keepdim=True)
q = model.eval_q(r, v) 

# get a trajectory
pred_traj = []
current_state = start_state
done = False
steps = 0
while not done and steps < world.n_states + 100: # should be able to do it in less than n states right.
    state_x, state_y = SparseMap.roomIndexToRc(env.grid, current_state)
    pred_traj.append((state_x, state_y))
    # print('current state', G.get_coords(current_state))
    logits, action = model.get_action(q, state_x, state_y) #
    action = action.detach().numpy()
    next_state = np.random.choice(range(world.n_states), p=world.T[action, current_state]) # next state based on action and current state
    observation = np.random.choice(range(len(world.observations)), p=world.O[action, next_state]) # um what are these observations/what do they mean...
    if next_state == goal_state:
        done = True
        pred_traj.append(SparseMap.roomIndexToRc(env.grid, next_state))
    current_state = next_state
    print(current_state)
    steps += 1

if done==False:
    print('failed maybe')