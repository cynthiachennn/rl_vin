import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from tqdm import tqdm
import argparse
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from model import VIN
from test import test

import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

def expert_move(image, row, col, r_move, c_move):
    new_row = max(0, min(row + r_move, image.shape[0] - 1))
    new_col = max(0, min(col + c_move, image.shape[1] - 1))
    if image[new_row, new_col] != image[row, col]:
        new_row = row
        new_col = col
    return new_row, new_col

def train(worlds, net, mode, config, epochs):
    log_datetime = str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    print(log_datetime)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'])
    device = config['device']
    batch_size = config['batch_size']

    # convert matrix to input view for model
    coords = np.empty((len(worlds), 4), dtype=int)
    for i, world in enumerate(worlds):
        start_x, start_y = (np.where(world == 3))
        goal_x, goal_y = (np.where(world == 2))
        coords[i] = np.array([start_x[0], start_y[0], goal_x[0], goal_y[0]])
        reward_mapping = -1 * np.ones(worlds.shape) # -1 for freespace
    reward_mapping[range(len(worlds)), coords[:, 2], coords[:, 3]] = 10
    grid_view = worlds.copy()
    grid_view[range(len(worlds)), coords[:, 0], coords[:, 1]] = 0
    grid_view[range(len(worlds)), coords[:, 2], coords[:, 3]] = 0
    grid_view = np.reshape(grid_view, (len(worlds), 1, worlds.shape[1], worlds.shape[2]))
    reward_view = np.reshape(reward_mapping, (len(worlds), 1, worlds.shape[1], worlds.shape[2]))
    worlds = np.concatenate((grid_view, reward_view), axis=1)

    if mode == 'expert':
        experts = np.empty((0, worlds.shape[1], worlds.shape[2]))
        for i, world in enumerate(worlds):
            P = np.zeros((world.shape[1] * world.shape[2], world.shape[1] * world.shape[2], config['n_act']), dtype=int)
            for row in range(world.shape[1]):
                for col in range(world.shape[2]):
                    curr_state = np.ravel_multi_index([row, col], (world.shape[1], world.shape[2]), order='F')
                    for i_action, action in enumerate(net.module.actions):
                        neighbor_row, neighbor_col = expert_move(worlds[i][0], row, col, action[0], action[1])
                        neighbor_state = np.ravel_multi_index([neighbor_row, neighbor_col], (world.shape[1], world.shape[2]), order='F')
                        P[curr_state, neighbor_state, i_action] = 1
            g_dense = np.logical_or.reduce(P, axis=2).T
            g_sparse = csr_matrix(g_dense)
            goal = np.ravel_multi_index([coords[i, 2], coords[i, 3]], (world.shape[1], world.shape[2]), order='F')
            _, pred = dijkstra(g_sparse, indices=goal, return_predecessors=True)
            temp_expert = np.zeros((world.shape[1] * world.shape[2]), dtype=int)
            for i in range(pred.shape[0]): # for each state
                next_state = pred[i]
                if next_state == -9999:
                    action = 4
                else:
                    action = np.argmax(P[i, next_state])
                temp_expert[i] = action
            temp_expert = np.reshape(temp_expert, (1, worlds[i].shape[1], worlds[i].shape[2]), order='F')
            experts = np.concatenate((experts, temp_expert), axis=0)
            experts = torch.tensor(experts, dtype=torch.float, device=device)

    worlds = torch.tensor(worlds, dtype=torch.float, device=device)
    coords = torch.tensor(coords, dtype=torch.int, device = device)
    
    save_train_loss = []
    for epoch in range(epochs):
        print('epoch:', epoch)
        explore_start = datetime.now()
        train_loss = 0.0
        total = worlds.shape[0]/batch_size
        for idx in range(0, len(worlds), batch_size):
            input_view = worlds[idx: idx + batch_size]
            batch_coords = coords[idx: idx + batch_size]
            
            if len(worlds) < batch_size:
                continue

            if mode == 'rl':
                data = torch.empty((2, 0, batch_size, config['n_act'])).to(device)
                values = net(input_view, batch_coords, mode)
                data = torch.cat((data, values), dim=1)
                pred, targ = data

            elif mode == 'expert':
                states_x = torch.empty((0, batch_size), dtype=torch.int64).to(device)
                states_y = torch.empty((0, batch_size), dtype=int).to(device)
                pred = torch.empty((0, batch_size)).to(device)
                target = torch.empty((0, batch_size), dtype=int).to(device)
                
                logits, trajectory = net(input_view, batch_coords, mode='expert')
                states_x = torch.cat((states_x, trajectory[:, :, 0]), dim=0)
                states_y = torch.cat((states_y, trajectory[:, :, 1]), dim=0)
                logits = torch.transpose(torch.transpose(logits, 1, 0), 2, 1)
                target = torch.transpose(target, 1, 0)
                pred, targ = logits, target

            optimizer.zero_grad()

            loss = criterion(pred, targ)
            train_loss += loss.item() / total
            loss.backward()
            optimizer.step()
        save_train_loss.append(train_loss)
        net.module.exploration_prob = max(net.module.exploration_prob * 0.99, 0.01)
        torch.save(net.state_dict(), f'saved_models/{log_datetime}_{'FINAL'}_{worlds.shape[2]}x{worlds.shape[3]}_{len(worlds)}_x{epochs}.pt')
        print('epoch time:', datetime.now() - explore_start)


def main(trainfiles, testfile, mode, epochs):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(device, torch.cuda.device_count())

    config = {
        "n_act": 5, 
        "lr": 0.001,
        'l_i': 2,
        'l_h': 150,
        "l_q": 10,
        "k": 20,
        "batch_size": 32,
        "max_steps": 50,
        "device": device,
    }

    net = VIN(config).to(device)
    net = torch.nn.DataParallel(net)


    if type(trainfiles) == str:
        trainfiles = [trainfiles]

    for i, file in enumerate(trainfiles):
        worlds = np.load(file)
        imsize = worlds[0].shape[0]
        config['max_steps'] = imsize * imsize * 2
        net.module.exploration_prob = 1 - (i * 0.1)
        train(worlds, net, mode, config, epochs)

    worlds_test = np.load(testfile)
    test(worlds_test, net)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainfiles', '-train', type=str, nargs='+', default='dataset/train_worlds/small_4_4_20000.npy')
    parser.add_argument('--testfile', '-test', default='dataset/test_worlds/small_4_4_2000.npy')
    parser.add_argument('--mode', '-m', default='rl')
    parser.add_argument('--epochs', '-e', default=400)
    args = parser.parse_args()
    
    main(args.trainfiles, args.testfile, args.mode, args.epochs)
