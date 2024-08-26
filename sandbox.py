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
    # else "mps"
    # if torch.backends.mps.is_available()
    else "cpu"
)

print(device)
print ("NUM:", torch.cuda.device_count())

# check multi gpu code in 
datafile = 'dataset/rl/small_4_4_1024.npz' # type_size_density_n_envs
with np.load(datafile) as f:
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
    "device": device,
    "n_act": 5, 
    "lr": 0.005,
    'l_i': 2,
    'l_h': 150,
    "l_q": 10,
    "k": 20,
}

model = VIN(config).to(device)
parallel_model = torch.nn.DataParallel(model)
model = parallel_model.module
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(parallel_model.parameters(), lr=config['lr'])

# will clean these up they are just random helper functions 
class Trajectories(Dataset):
    def __init__(self, inputView, memories, target_values):
        self.inputView = inputView
        self.memories = memories
        self.target_values = target_values
    
    def __len__(self):
        return len(self.memories)
    
    def __getitem__(self, idx):
        inputView = self.inputView[idx]
        memories = self.memories[idx]
        target_values = self.target_values[idx]
        return inputView, memories, target_values

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
    data = [[], [], []]
    for world in worlds_train:
        start_state = rng.integers(len(world.states)) # random start state idx
        goal_state = SparseMap.rcToRoomIndex(world.grid, world.goal_r, world.goal_c)        
        # would be nice to store this/method for this directly in world object ?
        n_traj = 4
        max_steps = 5000
        for traj in range(n_traj):
            inputViews, trajectory, target_values = model(world, start_state, max_steps)
            data[0].extend(inputViews)
            data[1].extend(trajectory)
            data[2].extend(target_values)
            # data[2].extend([target_values] * memories.shape[0]) 
            # store data as 1 complete trajectory full of multiple memories in each entry
            # can shuffle the trajectories and also the memories within the trajectory ? < but not sure if shuffling memories is gooood tbh
    print('explore time:', datetime.now() - explore_start)
    # print(len(data[0]), len(data[1]), len(data[2]))
    dataset = Trajectories(data[0], data[1], data[2])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) #, collate_fn=collate_tensor_fn) #, collate_fn=custom_collate_fn)
    # initialize the model for training after experience collecting is done
    # Training???
    train_start = datetime.now()
    for inputView, experience, target in tqdm(dataloader): # automatically shuffles and batches the data maybe
        inputView = inputView.to(device)
        experience = experience.to(int)
        # train/learn for each experience
        r, v = model.process_input(inputView)
        q = model.value_iteration(r, v)
        # experience[current state, action, reward, next_state, done]
        optimizer.zero_grad() # when to do this. now or in traj loops?
        state_x, state_y = batchedRoomIndexToRc(inputView[:, 0], experience[:, 0]) # should I directly store states as tuple coords? maybe.
        next_state_x, next_state_y = batchedRoomIndexToRc(inputView[:, 0], experience[:, 3])
        q_pred, _ = model.get_action(q, state_x, state_y)
        q_pred[:, experience[:, 1]] = target.to(torch.float).to(device) # to.device?
        loss = criterion(model.get_action(q, state_x, state_y)[0], q_pred)
        loss.backward()
        optimizer.step()
    exploration_prob = exploration_prob * 0.99 # no real basis for why this. i think ive seen .exp and other things
    print('training time:', datetime.now() - train_start)
    print('epoch time:', datetime.now() - explore_start)

current_datetime = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
save_path = f'saved_models/{current_datetime}_{imsize}x{imsize}_{len(worlds_train)}_x{epochs}.pt'
torch.save(parallel_model.state_dict(), save_path)
parallel_model.eval()

# test >>????
with torch.no_grad():
    correct = 0
    # create new testing env (will perform badly if trained on only one env tho duh)
    for world in worlds_test: 
        start_state = rng.integers(len(world.states)) # random start state idx
        goal_state = world.rcToRoomIndex(world.goal_r, world.goal_c)
        # create input view
        # learn world
        r, v = model.process_input(torch.Tensor(world.inputView.to(device)))
        q = model.value_iteration(r, v)

        # get a trajectory
        pred_traj = []
        current_state = start_state
        done = False
        steps = 0
        while not done and steps < len(world.states) + 20: # should be able to do it in less than n states right.
            state_x, state_y = world.roomIndexToRc(current_state)
            pred_traj.append((state_x, state_y))
            # print('current state', G.get_coords(current_state))
            logits, action = model.get_action(q, state_x, state_y) #
            action = action.cpu() # detach().numpy()
            next_state = rng.choice(range(len(world.states)), p=world.T[action, current_state]) # next state based on action and current state
            observation = rng.choice(range(len(world.observations)), p=world.O[action, next_state]) # um what are these observations/what do they mean...
            if next_state == goal_state:
                done = True
                pred_traj.append(world.roomIndexToRc(next_state))
            current_state = next_state
            print("state", current_state, "action", action)
            steps += 1
        if done == True:
            correct += 1
            # visualize world and values ? 
            fig, ax = plt.subplots()
            plt.imshow(world.grid.T, cmap='Greys')
            ax.plot(world.goal_r, world.goal_c, 'ro')
            fig, ax = plt.subplots()
            q_max = [[np.max(model.get_action(q, r, c)[0].cpu().detach().numpy()) for c in range(world.grid.shape[1])] for r in range(world.grid.shape[0])]
            plt.imshow(q_max, cmap='viridis')
            plt.show()

        if done==False:
            print('failed?')

print('accuracy:', correct/len(worlds_train))