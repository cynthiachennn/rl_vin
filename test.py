import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse

from model import VIN
from domains.Worlds import World

rng = np.random.default_rng()

import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

def load_model(datafile, model_path):
        
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    worlds = np.load(datafile)
    rng.shuffle(worlds)

    config = {
        "device": device,
        "n_act": 5, 
        "lr": 0.005,
        'l_i': 2,
        'l_h': 150,
        "l_q": 10,
        "k": 20,
        "max_steps": 50
    }

    net = VIN(config).to(device)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(model_path))
    net.eval()

    return worlds, net

def test(data, net):
    device = 'cuda' # module
    with torch.no_grad():
        correct = 0
        # create new testing env (will perform badly if trained on only one env tho duh)
        for world in data:
            # pick a random free state for the start state
            free = False
            while free == False:
                start_x, start_y = rng.integers(world.shape[0]), rng.integers(world.shape[1])
                if world[start_x, start_y] == 0:
                    free = True
            goal_x, goal_y = np.where(world == 2)
            coords = [start_x, start_y, goal_x[0], goal_y[0]] # torch.cat((start_x, start_y, goal_x, goal_y))

            # create input view
            reward_mapping = -1 * np.ones(world.shape) # -1 for freespace
            reward_mapping[goal_x, goal_y] = 20 # what value at goal? also if regenerating goals for each world then i'd need to redo this for each goal/trajectory.
            world[goal_x, goal_y] = 0 # remove goal from grid view
            grid_view = np.reshape(world, (1, 1, world.shape[0], world.shape[1]))
            reward_view = np.reshape(reward_mapping, (1, 1, world.shape[0], world.shape[1]))
            input_view = np.concatenate((grid_view, reward_view), axis=1) # inlc empty 1 dim
            
            input_view = torch.tensor(input_view, dtype=torch.float, device=device) #.to(device)
            coords = torch.tensor(coords, dtype=torch.int, device=device).reshape((1, 4)) # migh tbe inefficent since I make it a tensor earlier :| hm


            trajectory = net(input_view, coords, test=True) # max steps = size of world?
            # print(trajectory.shape)
            if trajectory[-1, :, 4] == 0:
                print('success!')
                correct += 1

                # print trajectory at least
                print('states:', trajectory[:, :, 0], trajectory[:, :, 1])
                print('actions:', trajectory[:, :, 2])

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
                
            # if trajectory[-1, :, 4] == 1:
                # print('failed?')
    print('accuracy:', correct/len(data))

def main(datafile, model_path):
    data, net = load_model(datafile, model_path)
    test(data, net)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', '-d', type=str, default='dataset/saved_worlds/small_4_4_64.npy')
    parser.add_argument('--model_path', '-m', type=str, default='saved_models/2024-08-30 14-37-42_6x6_51_x32.pt')
    args = parser.parse_args()
    main(args.datafile, args.model_path)