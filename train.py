import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime
# from tqdm import tqdm
import argparse

from model import VIN

rng = np.random.default_rng() # is this bad to do?


import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

def train(worlds_train, net, config, criterion, optimizer, epochs, batch_size):
    for epoch in range(epochs):
        print('epoch:', epoch)
        explore_start = datetime.now()
        device = 'cpu' #net.output_device # module
        data = torch.empty((2, 0, batch_size, config['n_act'])).to(device) # literally the only place i use config['n_act'].. do better !!! # [target/pred, n_experiences, batch_size, n_actions]
        for idx in range(0, len(worlds_train), batch_size):
            worlds = worlds_train[idx: idx + batch_size] # try batching for faster??
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
            reward_mapping[range(batch_size), coords[:, 2], coords[:, 3]] = 10 # what value at goal? also if regenerating goals for each world then i'd need to redo this for each goal/trajectory.
            grid_view = worlds.copy()
            grid_view[range(batch_size), coords[:, 2], coords[:, 3]] = 0 # remove goal from grid view
            grid_view = np.reshape(worlds, (batch_size, 1, worlds.shape[1], worlds.shape[2]))
            reward_view = np.reshape(reward_mapping, (batch_size, 1, worlds.shape[1], worlds.shape[2]))
            input_view = np.concatenate((grid_view, reward_view), axis=1) # inlc empty 1 dim
            
            input_view = torch.tensor(input_view, dtype=torch.float, device=device) #.to(device)
            coords = torch.tensor(coords, dtype=torch.int, device=device) #.to(device) #.to(int)

            n_traj = 4
            for traj in range(n_traj):
                values = net(input_view, coords)
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
        net.module.exploration_prob = net.module.exploration_prob * 0.99 # no real basis for why this. i think ive seen .exp and other things
        # net.exploration_prob = net.exploration_prob * 0.99
        print('training time:', datetime.now() - train_start)
        print('epoch time:', datetime.now() - explore_start)


def test(worlds_test, net):
    device = 'cpu' # module
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


            trajectory = net(input_view, coords, test=True) # max steps = size of world?
            net.module.config['max_steps'] = 50
            if trajectory[-1, :, 4] == 0:
                print('success!')
                correct += 1

                # print trajectory at least
            print('states:', trajectory[:, :, 0].flatten().tolist(), trajectory[:, :, 1].flatten().tolist())
            print('actions:', trajectory[:, :, 2].flatten().tolist())

                # # visualize world and values ? 
                # r, v = net.module.process_input(input_view)
                # q = net.module.value_iteration(r, v)

                # fig, ax = plt.subplots()
                # plt.imshow(world.T, cmap='Greys')
                # ax.plot(goal_x, goal_y, 'ro')
                # fig, ax = plt.subplots()
                # q_max = [[np.max(net.module.get_action(q, r, c)[0].cpu().detach().numpy()) for c in range(world.grid.shape[1])] for r in range(world.grid.shape[0])]
                # plt.imshow(q_max, cmap='viridis')
                # plt.show()
                
            if trajectory[-1, :, 4] == 1:
                print('failed?')
    print('accuracy:', correct/len(worlds_test))

    
def main(datafile, epochs, batch_size):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(device, torch.cuda.device_count())
    
    worlds = np.load(datafile)
    worlds.shape
    rng.shuffle(worlds)
    imsize = worlds[0].shape[0]
    # print('changed train size = 1; all data used for train')
    train_size = 0.8# # part of parameters (?) so should this even stay here?
    worlds_train = worlds[:int(len(worlds)*train_size)]
    worlds_test = worlds[int(len(worlds)*train_size):]

    config = {
        "n_act": 5, 
        "lr": 0.005,
        'l_i': 2,
        'l_h': 150,
        "l_q": 10,
        "k": 20,
        "max_steps": 50,
    }

    net = VIN(config).to(device)
    net = torch.nn.DataParallel(net)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'])

    train(worlds_train, net, config, criterion, optimizer, epochs, batch_size)

    current_datetime = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    save_path = f'saved_models/{current_datetime}_{imsize}x{imsize}_{len(worlds_train)}_x{epochs}.pt'
    torch.save(net.state_dict(), save_path)

    test(worlds_test, net)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', '-df', default='dataset/saved_worlds/small_4_0_256.npy')
    parser.add_argument('--epochs', '-e', default=32)
    parser.add_argument('--batch_size', '-b', default=32)
    args = parser.parse_args()
    
    main(args.datafile, args.epochs, args.batch_size)    
