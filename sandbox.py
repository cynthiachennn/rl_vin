import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime

from model import VIN

rng = np.random.default_rng()

# all parameters here
device = (
    "cuda:1"
    if torch.cuda.is_available()
    else "cpu"
)

config = {
    "n_act": 5, 
    "lr": 0.005,
    'l_i': 2,
    'l_h': 150,
    "l_q": 10,
    "k": 20,
    "max_steps": 50,
}
print(device, torch.cuda.device_count())
imsize = 4
n_worlds = 1024
epochs = 32
batch_size = 32

datafile = f'dataset/saved_worlds/small_{imsize}_4_{n_worlds}.npy'

worlds = np.load(datafile)
worlds.shape
imsize = worlds[0].shape[0]
# print('changed train size = 1; all data used for train')
train_size = 0.75# # part of parameters (?) so should this even stay here?
worlds_train = worlds[:int(len(worlds)*train_size)]
worlds_test = worlds[int(len(worlds)*train_size):]


self = VIN(config).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(self.parameters(), lr=config['lr'])

### begin TRAIN
plt.ion()

for epoch in range(epochs):
    print('epoch:', epoch)
    explore_start = datetime.now()
    device = 'cpu' #net.output_device # module
    for idx in range(0, len(worlds_train), batch_size):
        data = torch.empty((2, 0, batch_size, config['n_act'])).to(device) # literally the only place i use config['n_act'].. do better !!! # [target/pred, n_experiences, batch_size, n_actions]
        worlds = worlds_train[idx: idx + batch_size]
        if len(worlds) < batch_size:
            continue
        # pick a random free state for the start state
        
        coords = np.empty((batch_size, 4), dtype=int)
        for i in range(batch_size):
            free = False
            while free == False: 
                start_x, start_y = rng.integers(worlds.shape[1]), rng.integers(worlds.shape[2])
                if worlds[i, start_x, start_y] == 0:
                    free = True
            goal_x, goal_y = (np.where(worlds[i] == 2))
            temp = np.array([start_x, start_y, goal_x[0], goal_y[0]])
            coords[i] = temp
        
        # create input view
        reward_mapping = -1 * np.ones(worlds.shape) # -1 for freespace
        reward_mapping[coords[:, 2], coords[:, 3]] = 10 # what value at goal? also if regenerating goals for each world then i'd need to redo this for each goal/trajectory.
        grid_view = worlds.copy()
        grid_view[coords[:, 2], coords[:, 3]] = 0 # remove goal from grid view
        grid_view = np.reshape(worlds, (batch_size, 1, worlds.shape[1], worlds.shape[2]))
        reward_view = np.reshape(reward_mapping, (batch_size, 1, worlds.shape[1], worlds.shape[2]))
        input_view = np.concatenate((grid_view, reward_view), axis=1) # inlc empty 1 dim
        
        input_view = torch.tensor(input_view, dtype=torch.float, device=device) #.to(device)
        coords = torch.tensor(coords, dtype=torch.int, device=device) #.to(device) #.to(int)

        n_traj = 1
        test = False
        for traj in range(n_traj):
            # maybe track the evolution of q_value
            self.rng = np.random.default_rng() ## < idk
            state_x, state_y, goal_x, goal_y = torch.transpose(coords, 0, 1)
            grid = input_view[:, 0]
            device = 'cpu'  # input_view.get_device()   ### module

            trajectory = torch.empty(size=(0, batch_size, 5), dtype=torch.int, device=device)
            total_steps = 0
            done = torch.ones(batch_size, device=device)
            r, v = self.process_input(input_view)
            q = self.value_iteration(r, v)

            # # plot the value mapping i guess???
            # q_viz = torch.zeros((grid.shape[1] * 3, grid.shape[2] * 3))
            # for r in range(grid.shape[1]):
            #     for c in range(grid.shape[2]):
            #         qact = self.fc(q[:, :, r, c])[0]
            #         for action in range(config['n_act']):
            #             r_shift, c_shift = self.actions[action]
            #             q_viz[r * 3 + 1 + r_shift][c * 3 + 1 + c_shift] = qact[action]

            # # plot qvalues ?
            # q_viz = q_viz.detach().numpy()
            # fig, (ax1, ax2) = plt.subplots(1, 2)
            # fig.suptitle(f'epoch: {epoch}')
            # ax1.imshow(q_viz, cmap='viridis')
            # # plt.colorbar()
            # plt.show()

            # # plot grid
            # ax2.imshow(grid[0].T, cmap='Greys')
            # ax2.plot(state_x, state_y, 'ro')
            # ax2.plot(goal_x, goal_y, 'go')
            # plt.draw()
            # plt.waitforbuttonpress()
            # actiondict = ['right', 'up', 'left', 'down', 'stay']
            
            while torch.any(done == 1) and total_steps < self.config['max_steps']:
                if test is False:
                    if self.rng.random() < self.exploration_prob:
                        action = torch.tensor(self.rng.choice(len(self.actions), batch_size), device=device)
                    else:
                        logits, action = self.get_action(q, state_x, state_y)
                        # print(action)
                else: # test = true means always follow policy
                    logits, action = self.get_action(q, state_x, state_y)
                next_x, next_y, reward = self.move(input_view, action, state_x, state_y)
                # done = 0 means goal has been reached, weird notation because i need to disregard values after reaching the goal
                done = done * torch.where(done == 2, 0, 1) # if in the LAST STEP, the agent reached the goal, done = 2; that means from now ON, done = 0/mark that goal is previously reached
                done = done * torch.where(((next_x == goal_x) & (next_y == goal_y)), 2, 1) # # if we just reached the goal at this step, set done = 2 so we can write that down, and then in the next cycle done will be reset to 0
                experience = torch.vstack((state_x, state_y, action, reward, done)).transpose(0, 1).reshape((1, batch_size, 5))
                trajectory = torch.cat((trajectory, experience))
                
                # visualize the movement
                # ax2.plot(state_x, state_y, 'ko')
                # ax2.plot(next_x, next_y, 'ro')
                # print(actiondict[action[0]])
                # plt.show()
                # plt.waitforbuttonpress()
                
                state_x, state_y = next_x, next_y
                total_steps += 1
            
            trajectory = trajectory.to(torch.int) # shape: [batch_size, n_steps ,5]

            trajectory[:, :, 3] = trajectory[:, :, 3] * trajectory[:, :, 4] # set reward past done to 0 hopefully ?
            q_target = [torch.sum(trajectory[i:, range(batch_size), 3], dim=0) * 0.2 for i in range(trajectory.shape[0])]
            q_target = torch.stack(q_target).to(device).to(torch.float) # q_target.shape = [n_episodes, batch_size]
            pred_values = torch.empty((0, batch_size, len(self.actions)), device=device) # [episodes, batch_size, n_actions]
            for episode in trajectory:
                state_x, state_y = episode[range(batch_size), 0], episode[range(batch_size), 1]
                q_pred, _ = self.get_action(q, state_x, state_y)
                pred_values = torch.cat((pred_values, q_pred.unsqueeze(0)))
            target_values = torch.clone(pred_values)
            # target_values[range(target_values.shape[0]):, range(batch_size), trajectory[range(trajectory.shape[0]), range(batch_size), 2]] = torch.Tensor(q_target).to(device) # not sure if .to(device) is necessary
            # this part is weird chatgpt magic but i think it works...
            indices = trajectory[:, :, 2]
            batch_indices = torch.arange(target_values.size(0)).view(-1, 1).expand_as(indices)
            column_indices = torch.arange(target_values.size(1)).expand_as(indices)
            target_values[batch_indices, column_indices, indices] = q_target
            # einsum feels like black magic but I actually think this works/makes sense
            pred_values = torch.einsum('eba,eb->eba', [pred_values, trajectory[:, :, 4]])
            target_values = torch.einsum('eba,eb->eba', [target_values, trajectory[:, :, 4]]) 
            
            values = torch.stack((pred_values, target_values))

            # print trajectory information for some transparency i guess...

            data = torch.cat((data, values), dim=1)
            # store data as 1 complete trajectory full of multiple memories in each entry
            # can shuffle the trajectories and also the memories within the trajectory ? < but not sure if shuffling memories is gooood tbh
        # print('explore time:', datetime.now() - explore_start)

        train_start = datetime.now()
        optimizer.zero_grad() # start training
        # print(data[0].shape)
        # print(data[0].tolist())
        # print(data[1].tolist())
        loss = criterion(data[0], data[1])
        print(loss)
        loss.backward()
        optimizer.step()
    self.exploration_prob = self.exploration_prob * 0.99 # no real basis for why this. i think ive seen .exp and other things
    # net.exploration_prob = net.exploration_prob * 0.99
    print('training time:', datetime.now() - train_start)
    print('epoch time:', datetime.now() - explore_start)


current_datetime = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
save_path = f'saved_models/{current_datetime}_{imsize}x{imsize}_{len(worlds_train)}_x{epochs}.pt'
torch.save(self.state_dict(), save_path)


# # load data and model
# datafile = 'dataset/saved_worlds/small_4_4_64.npy'
# worlds_test= np.load(datafile)
# # model_path = 'saved_models/2024-09-03 16-53-17_6x6_6_x32.pt'
# model_path = 'saved_models/2024-08-30 17-44-34_6x6_204_x32.pt'
# self = VIN(config).to(device)
# self.load_state_dict(torch.load(model_path))
# self.eval()


### TEST
plt.ion()
with torch.no_grad():
    correct = 0
    # create new testing env (will perform badly if trained on only one env tho duh)
    for world in worlds_test:
        # pick a random free state for the start state
        free = False
        while free == False:
            start_x, start_y = rng.integers(world.shape[0]), rng.integers(world.shape[1])
            if world[start_x, start_y] == 0:
                free = True
        goal_x, goal_y = np.where(world == 2)
        coords = [start_x, start_y, goal_x[0], goal_y[0]] # torch.cat((start_x, start_y, goal_x, goal_y))

        # create input view
        reward_mapping = -0.1 * np.ones(world.shape) # -1 for freespace
        reward_mapping[goal_x, goal_y] = 1 # what value at goal? also if regenerating goals for each world then i'd need to redo this for each goal/trajectory.
        world[goal_x, goal_y] = 0 # remove goal from grid view
        grid_view = np.reshape(world, (1, 1, world.shape[0], world.shape[1]))
        reward_view = np.reshape(reward_mapping, (1, 1, world.shape[0], world.shape[1]))
        input_view = np.concatenate((grid_view, reward_view), axis=1) # inlc empty 1 dim
        
        input_view = torch.tensor(input_view, dtype=torch.float, device=device) #.to(device)
        coords = torch.tensor(coords, dtype=torch.int, device=device).reshape((1, 4)) # migh tbe inefficent since I make it a tensor earlier :| hm

        trajectory = self(input_view, coords, test=True) # max steps = size of world?
        self.config['max_steps'] = 50
        if trajectory[-1, :, 4] == 2:
            print('success!')
            correct += 1

        r, v = self.process_input(input_view)
        q = self.value_iteration(r, v)

        # plot the value mapping i guess???
        actiondict = ['right', 'up', 'left', 'down', 'stay']
        grid = input_view[:, 0]
        q_viz = torch.zeros((grid.shape[1] * 3, grid.shape[2] * 3))
        for r in range(grid.shape[1]):
            for c in range(grid.shape[2]):
                qact = self.fc(q[:, :, r, c])[0]
                for action in range(config['n_act']):
                    r_shift, c_shift = self.actions[action]
                    q_viz[r * 3 + 1 + r_shift][c * 3 + 1 + c_shift] = qact[action]
        
        # plot qvalues 
        q_viz = q_viz.detach().numpy()
        fig, (ax1, ax2) = plt.subplots(1, 2)
        # fig.suptitle(f'epoch: {epoch}')
        ax1.imshow(q_viz, cmap='viridis')
        # plt.colorbar()
        plt.show()

        ax2.imshow(grid[0].T, cmap='Greys')
        ax2.plot(start_x, start_y, 'ro')
        ax2.plot(goal_x, goal_y, 'go')
        plt.draw()
        plt.waitforbuttonpress()

        last_x, last_y = trajectory[0, 0, 0], trajectory[0, 0, 1]
        for state in trajectory:
            ax2.plot(last_x, last_y, 'ko')
            ax2.plot(state[:, 0], state[:, 1], 'ro')
            last_x, last_y = state[:, 0], state[:, 1]
            plt.draw()
        
        print('states:', trajectory[:, :, 0].flatten().tolist(), trajectory[:, :, 1].flatten().tolist())
        print('actions:', trajectory[:, :, 2].flatten().tolist())
        plt.waitforbuttonpress()
