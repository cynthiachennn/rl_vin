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
    device = 'cpu' # 'cuda' # module
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
            print('states:', trajectory[:, :, 0].flatten().tolist(), trajectory[:, :, 1].flatten().tolist())
            print('actions:', trajectory[:, :, 2].flatten().tolist())

            # visualize world and values ? 
            r, v = net.module.process_input(input_view)
            q = net.module.value_iteration(r, v)
            # print(net.module.q.weight)

            # fig, ax = plt.subplots()
            # q_max = [[np.max(net.module.get_action(q, r, c)[0].cpu().detach().numpy()) for c in range(world.shape[1])] for r in range(world.shape[0])]
            # plt.imshow(q_max, cmap='viridis')
            # plt.colorbar()

            # # want to sum q_values of adjacent states into current... is that not just a convolution lmao ? 
            # fig, ax = plt.subplots()
            # q_sum = torch.nn.functional.conv2d(q, torch.ones(1, 10, 3, 3), stride=1, padding=1)
            # # yes but actualy we don't want the sum we want to display max lol
            # q_cpu = np.pad(q.cpu().detach().numpy()[0, 0], 1)
            # q_sum = [np.max(q_cpu[i:i+3, j:j+3]) for i in range(6) for j in range(6)]
            # print(q_sum)

            fig, ax = plt.subplots()
            q_logits = [[net.module.get_action(q, r, c)[0].cpu().detach().numpy() for c in range(world.shape[1])] for r in range(world.shape[0])]
            q_logits = np.array(q_logits)
            new_matrix = np.zeros((8, 8, 5))
            for row in range(6):
                for col in range(6):
                    for act in range(5):
                        action = net.module.actions[act]
                        new_matrix[row + 1 + action[0], col + 1 + action[1], act] = q_logits[row, col, 0, act]
            q_sum = [np.max(new_matrix[i, j]) for i in range(1,7) for j in range(1,7)]
            plt.imshow(np.array(q_sum).reshape(6, 6), cmap='plasma')
            plt.colorbar()

            fig, ax = plt.subplots()
            plt.imshow(r[0, 0], cmap='Reds')    
            plt.colorbar()
            plt.show()

            fig, ax = plt.subplots()
            plt.imshow(world.T, cmap='Greys')
            ax.plot(start_x, start_y, 'bo')
            ax.plot(goal_x, goal_y, 'ro')
                
            # if trajectory[-1, :, 4] == 1:
                # print('failed?')
    print('accuracy:', correct/len(data))

def main(datafile, model_path):
    data, net = load_model(datafile, model_path)
    test(data, net)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', '-d', type=str, default='dataset/saved_worlds/small_4_0_256.npy')
    parser.add_argument('--model_path', '-m', type=str, default='saved_models/2024-09-03 14-46-04_6x6_204_x32.pt')
    args = parser.parse_args()
    main(args.datafile, args.model_path)