import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from tqdm import tqdm

from generators.sparse_map import  SparseMap
from dataset.generate_rl_dataset import SmallMap 
from model import VIN
from domains.batch_worlds import World

# ok in theory map gen goes in another script/ is loaded from a file but ill do that later.
# annoying things about the world I might wanna fix/change:
# only 4 actions + stay, not "in order"
# indexing is [a, s, s'] instead of [s, a, s']
# gridworld seems fancier with how it gets the transitions but this also makes more sense maybe...

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# SMALL MAPS
map_side_len = 4
obstacle_num = 4
num_envs = 4
discount = 0.99
envs = SmallMap.genMaps(num_envs, map_side_len, obstacle_num)
worlds = [World(env[0], env[1][0], env[1][1]) for env in envs]
# world grids directly from file into my new class

epochs = 50
batch_size = 128

config = {
    "imsize": map_side_len + 2, 
    "n_act": 5, 
    "lr": 0.005,
    'l_i': 2,
    'l_h': 150,
    "l_q": 10,
    "k": 20,
}

model = VIN(config).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

# will clean these up they are just random helper functions 
class Trajectories(Dataset):
    def __init__(self, memories, grid_view):
        self.memories = memories
        self.grid_view = grid_view
    
    def __len__(self):
        return len(self.memories)
    
    def __getitem__(self, idx):
        memories = self.memories[idx]
        grid_view = self.grid_view[idx]
        # print(len(memories))
        # print(len(grid_view))   
        # print(len(targets))
        return memories, grid_view

def collate_tensor_fn(batch): # look into why this is needed at some point... input_view shape is not consistent but idk why.
    mems = np.array([item[0] for item in batch])
    print([batch[i][1].shape for i in range(len(batch))])
    input_views = np.array([torch.squeeze(item[1], 0) for item in batch])
    return torch.Tensor(mems).to(int), torch.Tensor(input_views)

# batched ugh.
def batchedRoomIndexToRc(grid, room_index):
    room_rc = [torch.where(grid_i == 0) for grid_i in grid]
    room_r = []
    room_c = []
    for i in range(len(grid)):
        room_r.append(room_rc[i][0][int(room_index[i])])
        room_c.append(room_rc[i][1][int(room_index[i])])
    return room_r, room_c

exploration_prob = 1
for epoch in range(epochs):
    print('epoch:', epoch)
    explore_start = datetime.now()
    data = [[], []]
    count = 0
    for world in worlds:
        start_state = np.random.randint(len(world.states)) # random start state idx
        goal_state = SparseMap.rcToRoomIndex(world.grid, world.goal_r, world.goal_c)
        # create input view < do this before 
        reward_mapping = -1 * np.ones(world.grid.shape) # -1 for freespace
        reward_mapping[world.goal_r, world.goal_c] = 10 # 10 at goal, if regenerating goals for each world then i'd need to redo this for each goal/trajectory.
        grid_view = np.reshape(world.grid, (1, world.n_rows, world.n_cols))
        reward_view = np.reshape(reward_mapping, (1, world.n_rows, world.n_cols))
        input_view = torch.Tensor(np.concatenate((grid_view, reward_view))).to(device) # inlc empty 1 dim
        # would be nice to store this/method for this directly in world object ?
        n_traj = 20
        max_steps = 5000
        for traj in range(n_traj):
            trajectory_time = datetime.now()
            # should i regenerate the start and goal states each time?
            current_state = start_state
            memories = np.empty((0, 5), dtype=int) # hope dtype = int does not mess things up
            total_steps = 0
            done = 0
            # what should i name this function  o m g !
            r, v = model.process_input(input_view[None, :, :, :])
            for i in range(config['k'] - 1):
                q = model.eval_q(r, v)
                v, _ = torch.max(q, dim=1, keepdim=True)
            q = model.eval_q(r, v) 
            while done == 0 and total_steps < max_steps:
                step_time = datetime.now()
                state_x, state_y = world.roomIndexToRc(current_state)
                if np.random.rand() < exploration_prob:
                    action = np.random.choice(config['n_act']) # separate this from config... soon.
                else:
                    _, action = model.get_action(q, state_x, state_y)
                    action = action.cpu()
                next_state = np.random.choice(range(len(world.states)), p=world.T[action, current_state]) # next state based on action and current state
                observation = np.random.choice(range(len(world.observations)), p=world.O[action, next_state]) # um what are these observations/what do they mean...
                reward = world.R[action, current_state, next_state, observation]
                # end at goal
                if next_state == goal_state:
                    done = 1
                memories = np.vstack((memories, [current_state, action, reward, next_state, done]))
                current_state = next_state
                total_steps += 1
                count += 1

            # implement "experience replay" to propogate reward values
            target_values = [np.sum(memories[i:, 2]) for i in range(memories.shape[0])]
            memories = np.hstack((memories, np.array(target_values)[:, None])) #feels a little convoluted
            # WARNING: ^ this is later casted to int target can not be a float
            # if done == 1: # only store info if its successful
            data[0].extend([memories[i]for i in range(memories.shape[0])])
            data[1].extend([input_view] * memories.shape[0])
            # data[2].extend([target_values] * memories.shape[0]) 
            # store data as 1 complete trajectory full of multiple memories in each entry
            # can shuffle the trajectories and also the memories within the trajectory ? < but not sure if shuffling memories is gooood tbh
    print('explore time:', datetime.now() - explore_start)
    dataset = Trajectories(data[0], data[1])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) #, collate_fn=collate_tensor_fn) #, collate_fn=custom_collate_fn)
    # initialize the model for training after experience collecting is done
    # Trainign???
    train_start = datetime.now()
    for experience, input_view in tqdm(dataloader): # automatically shuffles and batches the data maybe
        input_view = input_view.to(device)
        experience = experience.to(int)
        r, v = model.process_input(input_view)
        for i in range(config['k'] - 1):
            q = model.eval_q(r, v)
            v, _ = torch.max(q, dim=1, keepdim=True)
        q = model.eval_q(r, v) 
        # train/learn for each experience
        # experience[current state, action, reward, next_state, done]
        optimizer.zero_grad() # when to do this. now or in traj loops?
        state_x, state_y = batchedRoomIndexToRc(input_view[:, 0], experience[:, 0]) # should I directly store states as tuple coords? maybe.
        next_state_x, next_state_y = batchedRoomIndexToRc(input_view[:, 0], experience[:, 3])
        q_pred, _ = model.get_action(q, state_x, state_y)
        q_target = experience[:, 5]
        q_pred[:, experience[:, 1]] = q_target.to(torch.float).to(device) # to.device?
        loss = criterion(model.get_action(q, state_x, state_y)[0], q_pred)
        loss.backward()
        optimizer.step()
    exploration_prob = exploration_prob * 0.95 # no real basis for why this. i think ive seen .exp and other things
    print('training time:', datetime.now() - train_start)
    print('epoch time:', datetime.now() - explore_start)


# test >>????
with torch.no_grad():
    # create new testing env (will perform badly if trained on only one env tho duh)
    env = SparseMap.genMaps(num_envs, map_side_len, obstacle_percent, scale)[0] # 0 = 0 freespace, 1 = obstacle
    world = World(env.grid, env.goal_r, env.goal_c)
    # start_state = np.random.randint(len(world.states)) # random start state idx
    # goal_state = SparseMap.rcToRoomIndex(env.grid, env.goal_r, env.goal_c)

    # create input view
    reward_mapping = -1 * np.ones(world.grid.shape) # -1 for freespace
    reward_mapping[world.goal_r, world.goal_c] = 10 # 10 at goal, if regenerating goals for each world then i'd need to redo this for each goal/trajectory.
    grid_view = np.reshape(world.grid, (1, 1, world.n_rows, world.n_cols))
    reward_view = np.reshape(reward_mapping, (1, 1, world.n_rows, world.n_cols))
    input_view = torch.Tensor(np.concatenate((grid_view, reward_view), axis=1)) # inlc empty 1 dim

    # would be nice to store this/method for this directly in world object
    # learn world
    r, v = model.process_input(input_view)
    for i in range(config['k'] - 1):
        q = model.eval_q(r, v)
        v, _ = torch.max(q, dim=1, keepdim=True)
    q = model.eval_q(r, v) 

    # visualize world and values ?
    fig, ax = plt.subplots()
    plt.imshow(world.grid.T, cmap='Greys')
    ax.plot(world.goal_r, world.goal_c, 'ro')
    fig, ax = plt.subplots()
    q_max = [[np.max(model.get_action(q, r, c)[0].detach().numpy()) for c in range(world.grid.shape[1])] for r in range(world.grid.shape[0])]
    plt.imshow(q_max, cmap='viridis')
    plt.show()
    # get a trajectory
    pred_traj = []
    current_state = start_state
    done = False
    steps = 0
    while not done and steps < len(world.states) + 100: # should be able to do it in less than n states right.
        state_x, state_y = SparseMap.roomIndexToRc(world.grid, current_state)
        pred_traj.append((state_x, state_y))
        # print('current state', G.get_coords(current_state))
        logits, action = model.get_action(q, state_x, state_y) #
        action = action.detach().numpy()
        next_state = np.random.choice(range(world.n_states), p=world.T[action, current_state]) # next state based on action and current state
        observation = np.random.choice(range(len(world.observations)), p=world.O[action, next_state]) # um what are these observations/what do they mean...
        if next_state == goal_state:
            done = True
            pred_traj.append(SparseMap.roomIndexToRc(world.grid, next_state))
        current_state = next_state
        print(current_state)
        steps += 1

    if done==False:
        print('failed?')