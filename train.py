import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from tqdm import tqdm
import argparse

from model import VIN
from test import test

rng = np.random.default_rng(9) # is this bad to do?

import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

def train(worlds, net, config, epochs, batch_size):
    log_datetime = str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    print(log_datetime)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'])
    device = net.output_device # module
    
    coords = np.empty((len(worlds), 4), dtype=int)
    for i in range(len(worlds)):
        start_x, start_y = (np.where(worlds[i] == 3))
        goal_x, goal_y = (np.where(worlds[i] == 2))
        coords[i] = np.array([start_x[0], start_y[0], goal_x[0], goal_y[0]])
        reward_mapping = -1 * np.ones(worlds.shape) # -1 for freespace
    reward_mapping[range(len(worlds)), coords[:, 2], coords[:, 3]] = 10 # what value at goal? also if regenerating goals for each world then i'd need to redo this for each goal/trajectory.
    grid_view = worlds.copy()
    grid_view[range(len(worlds)), coords[:, 0], coords[:, 1]] = 0 # remove start from grid view
    grid_view[range(len(worlds)), coords[:, 2], coords[:, 3]] = 0 # remove goal from grid view
    grid_view = np.reshape(grid_view, (len(worlds), 1, worlds.shape[1], worlds.shape[2]))  #unsqueeze(1)
    reward_view = np.reshape(reward_mapping, (len(worlds), 1, worlds.shape[1], worlds.shape[2]))
    worlds = np.concatenate((grid_view, reward_view), axis=1) # inlc empty 1 dim

    worlds = torch.tensor(worlds, dtype=torch.float, device=device)
    coords = torch.tensor(coords, dtype=torch.int, device = device)

    split = 0.8
    worlds_train = worlds[:int(len(worlds)*split)]
    coords_train = coords[:int(len(worlds)*split)]
    worlds_val = worlds[int(len(worlds)*split):]
    coords_val = coords[int(len(worlds)*split):]
    
    save_train_loss = []
    save_val_loss = []
    best_v_loss = 10000
    for epoch in range(epochs):
        print('epoch:', epoch)
        explore_start = datetime.now()
        train_loss = 0.0
        total = worlds_train.shape[0]/batch_size
        for idx in range(0, len(worlds_train), batch_size):
            data = torch.empty((2, 0, batch_size, config['n_act'])).to(device) # literally the only place i use config['n_act'].. do better !!! # [target/pred, n_experiences, batch_size, n_actions]
            input_view = worlds_train[idx: idx + batch_size] # try batching for faster??
            coords = coords_train[idx: idx + batch_size]
            if len(worlds) < batch_size:
                continue
            # pick a random free state for the start state
            
            n_traj = 1
            for traj in range(n_traj): # trajectory is same start.... so actually is there even a point in this?
                # or should i generate start here so i can get different starts for each trajectory?
                values = net(input_view, coords)
                data = torch.cat((data, values), dim=1)
                # store data as 1 complete trajectory full of multiple memories in each entry

            train_start = datetime.now()
            optimizer.zero_grad() # start training
            # idx = torch.randperm(len(data[0]))
            # data = data[:, idx]
            loss = criterion(data[0], data[1])
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
                if len(worlds) < batch_size:
                    continue
                values = net(input_view, coords)
                loss = criterion(values[0], values[1])
                val_loss += loss.item()/total_v
            save_val_loss.append(loss.item())
            print('epoch: ', epoch, 'train_loss: ', train_loss, 'val loss:', val_loss)
            if val_loss < best_v_loss:
                best_v_loss = val_loss
                print('save at epoch', epoch)
                torch.save(net.state_dict(), f'saved_models/{log_datetime}_{'VAL'}_{worlds.shape[2]}x{worlds.shape[3]}_{len(worlds_train)}.pt')

        torch.save(net.state_dict(), f'saved_models/{log_datetime}_{'FINAL'}_{worlds.shape[2]}x{worlds.shape[3]}_{len(worlds*0.8)}_x{epochs}.pt')
        print('epoch time:', datetime.now() - explore_start)
    np.save(f'loss/{log_datetime}_train_loss.npy', save_train_loss)
    np.save(f'loss/{log_datetime}_val_loss.npy', save_val_loss)


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
        "lr": 0.0001,
        'l_i': 2,
        'l_h': 150,
        "l_q": 10,
        "k": 20,
        "max_steps": imsize * imsize * 2,
    }

    net = VIN(config).to(device)
    net = torch.nn.DataParallel(net)

    train(worlds, net, config, epochs, batch_size)

    test(worlds_test, net, viz=False)

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainfile', '-train', default='dataset/train_worlds/small_4_4_20000x4.npy')
    parser.add_argument('--testfile', '-test', default='dataset/test_worlds/small_4_4_2000.npy')
    parser.add_argument('--epochs', '-e', default=400)
    parser.add_argument('--batch_size', '-b', default=32)
    args = parser.parse_args()
    
    main(args.trainfile, args.testfile, args.epochs, args.batch_size)    
