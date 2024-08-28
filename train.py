import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from tqdm import tqdm
import argparse

from model import VIN
from domains.Worlds import World

rng = np.random.default_rng() # is this bad to do?

def train(worlds_train, model, config, criterion, optimizer, epochs, batch_size):
    exploration_prob = 1
    for epoch in range(epochs):
        print('epoch:', epoch)
        explore_start = datetime.now()
        data = torch.empty((2, 0, config['n_act'])).to(config['device']) # stop using config['n_act'] !!!
        for world in worlds_train:
            # pick a random free state for the start state
            free = False
            while free == False: 
                start_x, start_y = rng.integers(world.shape[0]), rng.integers(world.shape[1])
                if world[start_x, start_y] == 0:
                    free = True
            goal_x, goal_y = np.where(world == 2)

            # create input view
            reward_mapping = -1 * np.ones(world.shape) # -1 for freespace
            reward_mapping[goal_x, goal_y] = 20 # what value at goal? also if regenerating goals for each world then i'd need to redo this for each goal/trajectory.
            world[goal_x, goal_y] = 0 # remove goal from grid view
            grid_view = np.reshape(world, (1, 1, world.shape[0], world.shape[1]))
            reward_view = np.reshape(reward_mapping, (1, 1, world.shape[0], world.shape[1]))
            input_view = np.concatenate((grid_view, reward_view), axis=1) # inlc empty 1 dim
            input_view = torch.Tensor(input_view)

            n_traj = 4
            max_steps = 5000
            for traj in range(n_traj):
                values = model(input_view, start_x, start_y, max_steps)
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


def test(worlds_test, model):
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

            # create input view
            reward_mapping = -1 * np.ones(world.shape) # -1 for freespace
            reward_mapping[goal_x, goal_y] = 20 # what value at goal? also if regenerating goals for each world then i'd need to redo this for each goal/trajectory.
            world[goal_x, goal_y] = 0 # remove goal from grid view
            grid_view = np.reshape(world, (1, 1, world.shape[0], world.shape[1]))
            reward_view = np.reshape(reward_mapping, (1, 1, world.shape[0], world.shape[1]))
            input_view = np.concatenate((grid_view, reward_view), axis=1) # inlc empty 1 dim
            input_view = torch.Tensor(input_view)

            trajectory = model(input_view, start_x, start_y, world[0]*world[1], test=True) # max steps = size of world?
            if trajectory[-1, 4] == 1:
                print('success!')
                correct += 1

                # print trajectory at least
                print('states:', trajectory[:, 0], trajectory[:, 1])
                print('actions:', trajectory[:, 2])

                # visualize world and values ? 
                r, v = model.module.process_input(input_view)
                q = model.module.value_iteration(r, v)

                fig, ax = plt.subplots()
                plt.imshow(world.T, cmap='Greys')
                ax.plot(goal_x, goal_y, 'ro')
                fig, ax = plt.subplots()
                q_max = [[np.max(model.module.get_action(q, r, c)[0].cpu().detach().numpy()) for c in range(world.grid.shape[1])] for r in range(world.grid.shape[0])]
                plt.imshow(q_max, cmap='viridis')
                plt.show()
                
            if trajectory[-1, 4] ==False:
                print('failed?')
    print('accuracy:', correct/len(worlds_test))

    
def main(datafile, epochs, batch_size):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    
    worlds = np.load(datafile)
    worlds.shape
    rng.shuffle(worlds)
    imsize = worlds[0].shape[0]
    train_size = 0.8 # part of parameters (?) so should this even stay here?
    worlds_train = worlds[:int(len(worlds)*train_size)]
    worlds_test = worlds[int(len(worlds)*train_size):]

    config = {
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

    train(worlds_train, model, config, criterion, optimizer, epochs, batch_size)

    current_datetime = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    save_path = f'saved_models/{current_datetime}_{imsize}x{imsize}_{len(worlds_train)}_x{epochs}.pt'
    torch.save(model.state_dict(), save_path)

    test(worlds_test, model)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', '-df', default='dataset/saved_worlds/small_4_4_64.npy')
    parser.add_argument('--epochs', '-e', default=32)
    parser.add_argument('--batch_size', '-b', default=32)
    args = parser.parse_args()
    
    main(args.datafile, args.epochs, args.batch_size)    
