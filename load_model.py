import numpy as np
import matplotlib.pyplot as plt
import torch

from model import VIN
from domains.Worlds import World

rng = np.random.default_rng()

device = (
    "cuda"
    if torch.cuda.is_available()
    # else "mps"
    # if torch.backends.mps.is_available()
    else "cpu"
)

data_file = 'dataset/rl/small_4_4_1024.npz' # type_size_density_n_envs
with np.load(data_file) as f:
    envs = f['arr_0']
    goals = f['arr_1']
imsize = envs.shape[1]
worlds = [World(envs[i], goals[i][0], goals[i][1]) for i in range(len(envs))]
rng.shuffle(worlds)
worlds_train = worlds[:int(0.8 * len(worlds))]
worlds_test = worlds[int(0.8 * len(worlds)):]

config = {
    "imsize": imsize, 
    "n_act": 5, 
    "lr": 0.005,
    'l_i': 2,
    'l_h': 150,
    "l_q": 10,
    "k": 20,
    "device": device,
}

model_path = 'saved_models/2024-08-26 17-36-27_6x6_819_x32.pt'

model = VIN(config).to(device)
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(model_path))
model.eval()

# test >>????with torch.no_grad():
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

            print(r.shape)

            print('states:', trajectory[:, 0])
            print('actions:', trajectory[:, 1])

            fig, ax = plt.subplots()
            plt.imshow(world.grid.T, cmap='Greys')
            ax.plot(world.goal_r, world.goal_c, 'ro')
            fig, ax = plt.subplots()
            q_max = [[np.max(model.module.get_action(q, r, c)[0].cpu().detach().numpy()) for c in range(world.grid.shape[1])] for r in range(world.grid.shape[0])]
            plt.imshow(q_max, cmap='viridis')
            fig, ax = plt.subplots()
            q_min = [[np.min(model.module.get_action(q, r, c)[0].cpu().detach().numpy()) for c in range(world.grid.shape[1])] for r in range(world.grid.shape[0])]
            plt.imshow(q_min, cmap='viridis') #fig 3
            plt.colorbar()
            fig, ax = plt.subplots()
            plt.imshow(r[0, 0], cmap='Reds')
            plt.colorbar()
            plt.show()
            
            # print trajectory at least
            


        if trajectory[-1, 4] ==False:
            print('failed?')

print('accuracy:', correct/len(worlds_test))