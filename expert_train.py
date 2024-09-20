from tracemalloc import start
import numpy as np
import matplotlib.pyplot as plt
from sympy import Q
import torch
from datetime import datetime
from tqdm import tqdm
import argparse
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


from model import VIN
from test import test

rng = np.random.default_rng(9) # is this bad to do?

import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0"


def trace_path(pred, source, target):
    # traces back shortest path from
    #  source to target given pred
    #  (a predicessor list)
    max_len = 1000
    path = np.zeros((max_len, 1))
    i = max_len - 1
    path[i] = target
    while path[i] != source and i > 0:
        try:
            path[i - 1] = pred[int(path[i])]
            i -= 1
        except Exception as e:
            return []
    if i >= 0:
        path = path[i:]
    else:
        path = None
    return path

def expert_move(image, row, col, r_move, c_move):
    new_row = max(0, min(row + r_move, image.shape[0] - 1))
    new_col = max(0, min(col + c_move, image.shape[1] - 1))
    if image[new_row, new_col] != image[row, col]:
        new_row = row
        new_col = col
    return new_row, new_col



def train(worlds, net, config, epochs, batch_size):
    log_datetime = str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    print(log_datetime)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'])
    device = 'cpu' #net.output_device # module

    # want the starts for validation to stay the same so i guess I gotta calculate it now ? 
    # also expert policy calculation
    coords = np.empty((len(worlds), 4), dtype=int)
    experts = np.empty((0, worlds.shape[1], worlds.shape[2]))
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
    worlds = np.concatenate((grid_view, reward_view), axis=1) # inlc empty 1 dim

    # create gridworld adjacency matrix, then run dijkstra on each to get expert policy
    for i in range(len(worlds)):
        P = np.zeros((worlds[i].shape[1] * worlds[i].shape[2], worlds[i].shape[1] * worlds[i].shape[2], config['n_act']), dtype=int)
        # for row in range(1, worlds[i].shape[1] - 1): # only ca;culate for inner square YIkES
        #     for col in range(1, worlds[i].shape[2] - 1):
        for row in range(worlds[i].shape[1]):
            for col in range(worlds[i].shape[2]):
                curr_state = np.ravel_multi_index([row, col], (worlds[i].shape[1], worlds[i].shape[2]), order='F')
                # print(curr_state)
                for i_action, action in enumerate(net.module.actions):
                    
                    neighbor_row, neighbor_col = expert_move(worlds[i][0], row, col, action[0], action[1])
                    neighbor_state = np.ravel_multi_index([neighbor_row, neighbor_col], (worlds[i].shape[1], worlds[i].shape[2]), order='F')
                    P[curr_state, neighbor_state, i_action] = 1     
        # Adjacency matrix of a graph connecting curr_state and next_state
        g_dense = np.logical_or.reduce(P, axis=2).T
        g_sparse = csr_matrix(g_dense)
        goal = np.ravel_multi_index([coords[i, 2], coords[i, 3]], (worlds.shape[2], worlds.shape[3]), order='F') # give waring about copy idk;
        start = np.ravel_multi_index([coords[i, 0], coords[i, 1]], (worlds.shape[2], worlds.shape[3]), order='F')
        d, pred = dijkstra(g_sparse, indices=goal, return_predecessors=True)
        # ok use pred to get action to shortest path but like how lol............
        # i guess its like go to state. check the pred for next state... then check which action leads to that state? okk that actually makes sense
        temp_expert = np.zeros((worlds[i].shape[1] * worlds[i].shape[2]), dtype=int)
        for i in range(pred.shape[0]): # for each state
            next_state = pred[i]
            if next_state == -9999:
                action = 4
            else:
                action = np.argmax(P[i, next_state]) # 
            temp_expert[i] = action
        temp_expert = np.reshape(temp_expert, (1, worlds[i].shape[1], worlds[i].shape[2]), order='F')
        experts = np.concatenate((experts, temp_expert), axis=0)

    worlds = torch.tensor(worlds, dtype=torch.float, device=device)
    coords = torch.tensor(coords, dtype=torch.int, device = device)
    experts = torch.tensor(experts, dtype=torch.int, device=device)

    split = 0.8
    worlds_train = worlds[:int(len(worlds)*split)]
    coords_train = coords[:int(len(worlds)*split)]
    experts_train = experts[:int(len(worlds)*split)]
    worlds_val = worlds[int(len(worlds)*split):]
    coords_val = coords[int(len(worlds)*split):]
    experts_val = experts[int(len(worlds)*split):]
    
    save_train_loss = []
    save_val_loss = []
    best_v_loss = 10000
    for epoch in range(epochs):
        print('epoch:', epoch)
        explore_start = datetime.now()
        train_loss = 0.0
        total = worlds_train.shape[0]/batch_size
        for idx in range(0, len(worlds_train), batch_size):
            input_view = worlds_train[idx: idx + batch_size] # try batching for faster??
            coords = coords_train[idx: idx + batch_size]
            experts = experts_train[idx: idx + batch_size]

            states_x = torch.empty((0, batch_size), dtype=torch.int64).to(device)
            states_y = torch.empty((0, batch_size), dtype=int).to(device)
            pred = torch.empty((0, batch_size)).to(device)
            target = torch.empty((0, batch_size), dtype=int).to(device)
            if len(worlds) < batch_size:
                continue
            
            n_traj = 1
            for traj in range(n_traj): # trajectory is same start.... so actually is there even a point in this?
                logitsList, trajectory = net(input_view, coords, test=True)
                states_x = torch.cat((states_x, trajectory[:, :, 0]), dim=0)
                states_y = torch.cat((states_y, trajectory[:, :, 1]), dim=0)
                # pred = torch.cat((pred, trajectory[:, :, 3]), dim=0)
                # store data as 1 complete trajectory full of multiple memories in each entry
            for state in zip(states_x, states_y):
                target = torch.cat((target, experts[range(batch_size), state[0], state[1]].unsqueeze(0)))
            
            train_start = datetime.now()
            optimizer.zero_grad()
            # logitsList = torch.flatten(logitsList, start_dim=0, end_dim=1)
            # target = torch.flatten(target, start_dim=0, end_dim=1)

            logitsList = torch.transpose(torch.transpose(logitsList, 1, 0), 2, 1)
            target = torch.transpose(target, 1, 0)
            loss = criterion(logitsList, target)
            train_loss += loss.item() / total
            loss.backward()
            optimizer.step()
        save_train_loss.append(train_loss)
        print('explore time:', datetime.now() - explore_start)
        net.module.exploration_prob = max(net.module.exploration_prob * 0.99, 0.01) # no real basis for why this. i think ive seen .exp and other things

        val_loss = 0.0
        total_v = worlds_val.shape[0]/batch_size
        with torch.no_grad():
            for batch in range(0, len(worlds_val), batch_size):
                worlds = worlds_val[batch: batch + batch_size]
                coords = coords_val[batch: batch + batch_size]
                experts = experts_val[batch: batch + batch_size]

                states_x = torch.empty((0, batch_size), dtype=int).to(device)
                states_y = torch.empty((0, batch_size), dtype=int).to(device)
                pred = torch.empty((0, batch_size)).to(device)
                target = torch.empty((0, batch_size), dtype=int).to(device)

                if len(worlds) < batch_size:
                    continue
                logitsList, trajectory = net(input_view, coords, test=True)

                states_x = torch.cat((states_x, trajectory[:, :, 0]), dim=0)
                states_y = torch.cat((states_y, trajectory[:, :, 1]), dim=0)
                pred = torch.cat((pred, trajectory[:, :, 3]), dim=0)
                # store data as 1 complete trajectory full of multiple memories in each entry
                for state in zip(states_x, states_y):
                    target = torch.cat((target, experts[range(batch_size), state[0], state[1]].unsqueeze(0)))

                logitsList = torch.transpose(torch.transpose(logitsList, 1, 0), 2, 1)
                target = torch.transpose(target, 1, 0)
                
                loss = criterion(logitsList, target)
                val_loss += loss.item()/total_v
            save_val_loss.append(loss.item())
            print('epoch: ', epoch, 'train_loss: ', train_loss, 'val loss:', val_loss)
            if val_loss < best_v_loss:
                best_v_loss = val_loss
                print('save at epoch', epoch)
                # torch.save(net.state_dict(), f'saved_models/{log_datetime}_{'VAL'}_{worlds.shape[2]}x{worlds.shape[3]}_{len(worlds_train)}.pt')

        torch.save(net.state_dict(), f'saved_models/expert/{log_datetime}_{'FINAL'}_{worlds.shape[2]}x{worlds.shape[3]}_{len(worlds*0.8)}_x{epochs}.pt')
        print('epoch time:', datetime.now() - explore_start)
    # np.save(f'loss/{log_datetime}_train_loss.npy', save_train_loss)
    # np.save(f'loss/{log_datetime}_val_loss.npy', save_val_loss)



def main(trainfile, testfile, epochs, batch_size):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(device, torch.cuda.device_count())
    
    worlds = np.load(trainfile)
    imsize = worlds[0].shape[0]
    worlds_test = np.load(testfile)

    config = {
        "n_act": 5, 
        "lr": 0.001,
        'l_i': 2,
        'l_h': 150,
        "l_q": 10,
        "k": 20,
        "max_steps": 50,
    }

    net = VIN(config).to(device)
    net = torch.nn.DataParallel(net)

    train(worlds, net, config, epochs, batch_size)

    test(worlds_test, net, viz=False)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainfile', '-train', default='dataset/train_worlds/sparse_16_20_20000x4.npy')
    # parser.add_argument('--trainfile', '-train', default='dataset/test_worlds/small_4_4_2000.npy')
    parser.add_argument('--testfile', '-test', default='dataset/test_worlds/sparse_16_20_2000.npy')
    parser.add_argument('--epochs', '-e', default=250)
    parser.add_argument('--batch_size', '-b', default=32)
    args = parser.parse_args()
    
    main(args.trainfile, args.testfile, args.epochs, args.batch_size)    
