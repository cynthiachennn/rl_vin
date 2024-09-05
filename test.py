import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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

def test(data, net, viz):
    device = 'cpu' # 'cuda' # module
    with torch.no_grad():
        correct = 0
        pathlengths = []
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
            if trajectory[-1, :, 4] == 2:
                print('success!')
                pathlengths.append(len(trajectory))
                correct += 1

            if viz:
                print('states:', [item for item in zip(trajectory[:, :, 0].flatten().tolist(), trajectory[:, :, 1].flatten().tolist())])
                print('actions:', trajectory[:, :, 2].flatten().tolist())
            
                # visualize world and values ? 
                r, v = net.module.process_input(input_view)
                q = net.module.value_iteration(r, v)

                # fig, ax = plt.subplots()
                # q_max = [[np.max(net.module.get_action(q, r, c)[0].cpu().detach().numpy()) for c in range(world.shape[1])] for r in range(world.shape[0])]
                # plt.imshow(q_max, cmap='viridis')
                # plt.colorbar()

                fig, ax = plt.subplots()
                plt.imshow(r[0, 0].T, cmap='Reds')
                plt.colorbar()


                # for i in range(10):
                #     fig, ax = plt.subplots()
                #     plt.imshow(net.module.q.weight[i, 0], cmap='Greens')
                #     plt.colorbar()

                # # show max of q val in actual spot ??
                # fig, ax = plt.subplots()
                # q_logits = [[net.module.get_action(q, r, c)[0].cpu().detach().numpy() for c in range(world.shape[1])] for r in range(world.shape[0])]
                # q_logits = np.array(q_logits)
                # new_matrix = np.zeros((8, 8, 5))
                # for row in range(6):
                #     for col in range(6):
                #         for act in range(5):
                #             action = net.module.actions[act]
                #             new_matrix[row + 1 + action[0], col + 1 + action[1], act] = q_logits[row, col, 0, act]
                # q_sum = [np.max(new_matrix[i, j]) for i in range(1,7) for j in range(1,7)]
                # plt.imshow(np.array(q_sum).reshape(6, 6), cmap='plasma')
                # plt.colorbar()
                plt.ion()
                # plot the value mapping i guess???
                actiondict = ['right', 'up', 'left', 'down', 'stay']
                fig, (ax1, ax2) = plt.subplots(1, 2)

                grid = input_view[:, 0]
                q_viz = torch.zeros((grid.shape[1] * 3, grid.shape[2] * 3))
                maxes = [[], []]
                for r in range(grid.shape[1]):
                    for c in range(grid.shape[2]):
                        qact = net.module.fc(q[:, :, r, c])[0]
                        maxact = torch.argmax(qact)
                        for action in range(net.module.config['n_act']):
                            r_shift, c_shift = net.module.actions[action]
                            q_viz[r * 3 + 1 + r_shift][c * 3 + 1 + c_shift] = qact[action]
                            if action == maxact:
                                ax1.add_patch(Rectangle((r * 3 + 1 + r_shift - 0.5, c * 3 + 1 + c_shift - 0.5), 1, 1, fill=False, edgecolor='red'))
                                maxes[0].append(r * 3 + 1 + r_shift)
                                maxes[1].append(c * 3 + 1 + c_shift)

                # plot qvalues 
                q_viz = q_viz.detach().numpy()
                
                # fig.suptitle(f'epoch: {epoch}')
                ax1.imshow(q_viz.T, cmap='viridis')
                # plt.colorbar()
                plt.show()

                ax2.imshow(grid[0].T, cmap='Greys')
                ax2.plot(start_x, start_y, 'ro')
                ax2.plot(goal_x, goal_y, 'go')
                ax1.plot(start_x * 3 + 1, start_y * 3 + 1, 'ro')
                ax1.plot(goal_x * 3 + 1, goal_y * 3 + 1, 'go')
                plt.draw()
                plt.waitforbuttonpress()

                last_x, last_y = trajectory[0, 0, 0], trajectory[0, 0, 1]
                for state in trajectory:
                    ax2.plot(last_x, last_y, 'ko')
                    ax2.plot(state[:, 0], state[:, 1], 'ro')
                    last_x, last_y = state[:, 0], state[:, 1]
                    plt.draw()
                # if trajectory[-1, :, 4] == 1:
                    # print('failed?')
    print('accuracy:', correct/len(data))
    print('avg path length:', np.mean(pathlengths))
    

def main(datafile, model_path, viz):
    data, net = load_model(datafile, model_path)
    test(data, net, viz)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', '-d', type=str, default='dataset/test_worlds/small_6_6_1024.npy')
    parser.add_argument('--model_path', '-m', type=str, default='saved_models/2024-09-04 12-20-54_8x8_4096_x32.pt')
    parser.add_argument('--viz', '-v', action='store_true')
    args = parser.parse_args()
    main(args.datafile, args.model_path, args.viz)