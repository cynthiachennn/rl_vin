import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
import argparse

from model import VIN
from domains.Worlds import World

rng = np.random.default_rng(9)

import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

def load_model(datafile, model_path):
        
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    worlds = np.load(datafile)
    # rng.shuffle(worlds)

    config = {
        "device": device,
        "n_act": 5, 
        "lr": 0.001,
        'l_i': 2,
        'l_h': 150,
        "l_q": 10,
        "k": 20,
        "max_steps": 50
    }

    net = VIN(config).to(device)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    return worlds, net

def test(worlds, net, viz):
    device = 'cpu' # 'cuda' # module
    with torch.no_grad():
        correct = 0
        success_distance = []
        average_distance = []
        # create new testing env (will perform badly if trained on only one env tho duh)
        coords = np.empty((len(worlds), 4), dtype=int)
        
        for i in range(len(worlds)):
            start_x, start_y = (np.where(worlds[i] == 3))
            goal_x, goal_y = (np.where(worlds[i] == 2))

            temp = np.array([start_x[0], start_y[0], goal_x[0], goal_y[0]])
            coords[i] = temp
            reward_mapping = -1 * np.ones(worlds.shape) # -1 for freespace
        reward_mapping[range(len(worlds)), coords[:, 2], coords[:, 3]] = 10 # what value at goal? also if regenerating goals for each world then i'd need to redo this for each goal/trajectory.
        grid_view = worlds.copy()
        grid_view[range(len(worlds)), coords[:, 0], coords[:, 1]] = 0 # remove start from grid view
        grid_view[range(len(worlds)), coords[:, 2], coords[:, 3]] = 0 # remove goal from grid view
        grid_view = np.reshape(grid_view, (len(worlds), 1, worlds.shape[1], worlds.shape[2]))
        reward_view = np.reshape(reward_mapping, (len(worlds), 1, worlds.shape[1], worlds.shape[2]))
        input_view = np.concatenate((grid_view, reward_view), axis=1) # inlc empty 1 dim

        input_view = torch.tensor(input_view, dtype=torch.float, device=device)
        coords = torch.tensor(coords, dtype=torch.int, device=device)

        for world in zip(input_view, coords):
            input_view = world[0].unsqueeze(0)
            coords = world[1].unsqueeze(0)

            start_x, start_y, goal_x, goal_y = coords[0]

            trajectory = net(input_view, coords, test=True) # max steps = size of world?
            # print(trajectory.shape)
            if trajectory[-1, :, 4] == 2:
                #print('success!')
                success_distance.append(len(trajectory))
                correct += 1
            average_distance.append(abs(goal_x-start_x + goal_y-start_y))

            if viz:
                print('states:', [item for item in zip(trajectory[:, :, 0].flatten().tolist(), trajectory[:, :, 1].flatten().tolist())])
                print('actions:', trajectory[:, :, 2].flatten().tolist())
            
                # visualize world and values ? 
                r, v = net.module.process_input(input_view)
                q = net.module.value_iteration(r, v)

                # fig, ax = plt.subplots()
                # q_max = [[np.max(net.module.get_action(q, r, c)[0].cpu().detach().numpy()) for c in range(world.shape[1])] for r in range(world.shape[0])]
                # plt.imshow(q_max, cmap='viridis')
                # plt.colorb 
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
                # ax2.plot(start_x, start_y, 'ro')
                # ax2.plot(goal_x, goal_y, 'go')
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
    print('accuracy:', correct/len(worlds))
    print('success path length:', np.mean(success_distance))
    print('average path length:', np.mean(average_distance))
    

def main(datafile, model_path, viz):
    data, net = load_model(datafile, model_path)
    print(datafile)
    print(model_path)
    test(data, net, viz)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', '-d', type=str, default='dataset/test_worlds/sparse_8_20_2000.npy')
    parser.add_argument('--model_path', '-m', type=str, default='saved_models/2024-09-12-01-56-04_FINAL_10x10_32_x200.pt')
    parser.add_argument('--viz', '-v', action='store_true')
    args = parser.parse_args()
    main(args.datafile, args.model_path, args.viz)