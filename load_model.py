import numpy as np
import matplotlib.pyplot as plt
import torch

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

config = {
    "imsize": imsize, 
    "n_act": 5, 
    "lr": 0.005,
    'l_i': 2,
    'l_h': 150,
    "l_q": 10,
    "k": 20,
}

model_path = 'saved_models/4x4_64env_4traj_32e.pt'

model = VIN(config).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# test >>????
with torch.no_grad():
    correct = 0
    # create new testing env (will perform badly if trained on only one env tho duh)
    for world in worlds_test: 
        start_state = rng.integers(len(world.states)) # random start state idx
        goal_state = world.rcToRoomIndex(world.goal_r, world.goal_c)
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
            print(current_state)
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