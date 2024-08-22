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

rng = np.random.default_rng()

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

data_file = 'dataset/rl/small_4_4_64.npz' # type_size_density_n_envs
with np.load(data_file) as f:
    envs = f['arr_0']
    goals = f['arr_1']
imsize = envs.shape[1]
worlds = [World(envs[i], goals[i][0], goals[i][1]) for i in range(len(envs))]
rng.shuffle(worlds)
worlds_train = worlds[:int(0.8 * len(worlds))]
worlds_test = worlds[int(0.8 * len(worlds)):]

epochs = 32
batch_size = 32

config = {
    "imsize": imsize, 
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
        return memories, grid_view

# wheres a good place to put this lol. sparsemap class as a static function ?
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
    for world in worlds_train:
        start_state = rng.integers(len(world.states)) # random start state idx
        goal_state = SparseMap.rcToRoomIndex(world.grid, world.goal_r, world.goal_c)
        # create input view < do this before 
        reward_mapping = -1 * np.ones(world.grid.shape) # -1 for freespace
        reward_mapping[world.goal_r, world.goal_c] = 10 # 10 at goal, if regenerating goals for each world then i'd need to redo this for each goal/trajectory.
        grid_view = np.reshape(world.grid, (1, world.n_rows, world.n_cols))
        reward_view = np.reshape(reward_mapping, (1, world.n_rows, world.n_cols))
        input_view = torch.Tensor(np.concatenate((grid_view, reward_view))).to(device) # inlc empty 1 dim
        # would be nice to store this/method for this directly in world object ?
        n_traj = 4
        max_steps = 5000
        for traj in range(n_traj):
            trajectory_time = datetime.now()
            # should i regenerate the start and goal states each time?
            current_state = start_state
            memories = np.empty((0, 5), dtype=int) # hope dtype = int does not mess things up
            total_steps = 0
            done = 0
            r, v = model.process_input(input_view[None, :, :, :])
            q = model.value_iteration(r, v)
            while done == 0 and total_steps < max_steps:
                step_time = datetime.now()
                state_x, state_y = world.roomIndexToRc(current_state)
                if rng.random() < exploration_prob:
                    action = rng.choice(config['n_act']) # separate this from config... soon.
                else:
                    _, action = model.get_action(q, state_x, state_y)
                    action = action.cpu()
                next_state = rng.choice(range(len(world.states)), p=world.T[action, current_state]) # next state based on action and current state
                observation = rng.choice(range(len(world.observations)), p=world.O[action, next_state]) # um what are these observations/what do they mean...
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
    # Training???
    train_start = datetime.now()
    for experience, input_view in tqdm(dataloader): # automatically shuffles and batches the data maybe
        input_view = input_view.to(device)
        experience = experience.to(int)
        r, v = model.process_input(input_view)
        q = model.value_iteration(r, v)
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
    exploration_prob = exploration_prob * 0.99 # no real basis for why this. i think ive seen .exp and other things
    print('training time:', datetime.now() - train_start)
    print('epoch time:', datetime.now() - explore_start)

current_datetime = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
save_path = f'saved_models/{current_datetime}_{imsize}x{imsize}_{len(worlds_train)}_x{epochs}.pt'
torch.save(model.state_dict(), save_path)
model.eval()

# test >>????
with torch.no_grad():
    correct = 0
    # create new testing env (will perform badly if trained on only one env tho duh)
    for world in worlds_test: 
        start_state = rng.integers(len(world.states)) # random start state idx
        goal_state = world.rcToRoomIndex(world.grid, world.goal_r, world.goal_c)
        # create input view
        reward_mapping = -1 * np.ones(world.grid.shape) # -1 for freespace
        reward_mapping[world.goal_r, world.goal_c] = 10 # 10 at goal, if regenerating goals for each world then i'd need to redo this for each goal/trajectory.
        grid_view = np.reshape(world.grid, (1, 1, world.n_rows, world.n_cols))
        reward_view = np.reshape(reward_mapping, (1, 1, world.n_rows, world.n_cols))
        input_view = torch.Tensor(np.concatenate((grid_view, reward_view), axis=1)).to(device) # inlc empty 1 dim

        # learn world
        r, v = model.process_input(input_view)
        q = model.value_iteration(r, v)

        # get a trajectory
        pred_traj = []
        current_state = start_state
        done = False
        steps = 0
        while not done and steps < len(world.states) + 100: # should be able to do it in less than n states right.
            state_x, state_y = world.roomIndexToRc(world.grid, current_state)
            pred_traj.append((state_x, state_y))
            # print('current state', G.get_coords(current_state))
            logits, action = model.get_action(q, state_x, state_y) #
            action = action.cpu() # .detach().numpy()
            next_state = rng.choice(range(world.n_states), p=world.T[action, current_state]) # next state based on action and current state
            observation = rng.choice(range(len(world.observations)), p=world.O[action, next_state]) # um what are these observations/what do they mean...
            if next_state == goal_state:
                done = True
                pred_traj.append(world.roomIndexToRc(world.grid, next_state))
            current_state = next_state
            print(current_state)
            steps += 1
        if done == True:
            correct += 1
            # visualize world and values ? 
            fig, ax = plt.subplots()
            plt.imshow(world.grid.T, cmap='Greys')
            ax.plot(world.goal_r, world.goal_c, 'ro')
            fig, ax = plt.subplots()
            q_max = [[np.max(model.get_action(q, r, c)[0].detach().numpy()) for c in range(world.grid.shape[1])] for r in range(world.grid.shape[0])]
            plt.imshow(q_max, cmap='viridis')
            plt.show()

        if done==False:
            print('failed?')

print('accuracy:', correct/len(worlds_train))