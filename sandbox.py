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
model = torch.nn.DataParallel(model)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

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

exploration_prob = 1
for epoch in range(epochs):
    print('epoch:', epoch)
    explore_start = datetime.now()
    data = torch.empty((2, 0, config['n_act'])).to(device) # stop using config['n_act'] !!!
    for world in worlds_train:
        start_state = rng.integers(len(world.states)) # random start state idx
        goal_state = SparseMap.rcToRoomIndex(world.grid, world.goal_r, world.goal_c)        
        # would be nice to store this/method for this directly in world object ?
        n_traj = 4
        max_steps = 5000
        for traj in range(n_traj):
            values = model(world, start_state, max_steps)
            data = torch.cat((data, values), dim=1)
            # store data as 1 complete trajectory full of multiple memories in each entry
            # can shuffle the trajectories and also the memories within the trajectory ? < but not sure if shuffling memories is gooood tbh
    print('explore time:', datetime.now() - explore_start)

    train_start = datetime.now()
    optimizer.zero_grad() # start training
    idx = torch.randperm(len(data[0]))
    data = data[:, idx]
    loss = criterion(data[0], data[1])
    loss.backward()
    optimizer.step()
    model.module.exploration_prob = model.module.exploration_prob * 0.99 # no real basis for why this. i think ive seen .exp and other things
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
        goal_state = world.rcToRoomIndex(world.goal_r, world.goal_c)
        # create input view
        trajectory = model(world, start_state, len(world.states), test=True)
        if trajectory[-1, 4] == 1:
            print('success!')
            correct += 1
            # visualize world and values ? 
            r, v = model.module.process_input(torch.Tensor(world.inputView))
            q = model.module.value_iteration(r, v)

            fig, ax = plt.subplots()
            plt.imshow(world.grid.T, cmap='Greys')
            ax.plot(world.goal_r, world.goal_c, 'ro')
            fig, ax = plt.subplots()
            q_max = [[np.max(model.module.get_action(q, r, c)[0].cpu().detach().numpy()) for c in range(world.grid.shape[1])] for r in range(world.grid.shape[0])]
            plt.imshow(q_max, cmap='viridis')
            plt.show()

        if trajectory[-1, 4] ==False:
            print('failed?')

print('accuracy:', correct/len(worlds_train))