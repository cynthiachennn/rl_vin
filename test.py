import numpy as np
import torch
import argparse

from model_nofc import VIN

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
        "n_act": 5, 
        "lr": 0.001,
        'l_i': 2,
        'l_h': 150,
        "l_q": 10,
        "k": 20,
        "max_steps": 50,
        "device": device,
    }

    net = VIN(config).to(device)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    return worlds, net

def test(worlds, net):
    device = 'cpu' # 'cuda' # module
    net.config['max_steps'] = worlds.shape[1] * worlds.shape[2]
    with torch.no_grad():
        correct = 0
        success_distance = []
        average_distance = []
        coords = np.empty((len(worlds), 4), dtype=int)
        
        for i, world in enumerate(worlds):
            start_x, start_y = (np.where(world == 3))
            goal_x, goal_y = (np.where(world == 2))
            temp = np.array([start_x[0], start_y[0], goal_x[0], goal_y[0]])
            coords[i] = temp
            reward_mapping = -1 * np.ones(worlds.shape)
        reward_mapping[range(len(worlds)), coords[:, 2], coords[:, 3]] = 10
        grid_view = worlds.copy()
        grid_view[range(len(worlds)), coords[:, 0], coords[:, 1]] = 0
        grid_view[range(len(worlds)), coords[:, 2], coords[:, 3]] = 0
        grid_view = np.reshape(grid_view, (len(worlds), 1, worlds.shape[1], worlds.shape[2]))
        reward_view = np.reshape(reward_mapping, (len(worlds), 1, worlds.shape[1], worlds.shape[2]))
        input_view = np.concatenate((grid_view, reward_view), axis=1)

        input_view = torch.tensor(input_view, dtype=torch.float, device=device)
        coords = torch.tensor(coords, dtype=torch.int, device=device)

        for world in zip(input_view, coords):
            input_view = world[0].unsqueeze(0)
            coords = world[1].unsqueeze(0)

            start_x, start_y, goal_x, goal_y = coords[0]

            trajectory = net(input_view, coords, mode='test') 
            if trajectory[-1, :, 4] == 2:
                success_distance.append(len(trajectory))
                correct += 1
            average_distance.append(abs(goal_x-start_x + goal_y-start_y))

            
    print('accuracy:', correct/len(worlds))
    print('success path length:', np.mean(success_distance))
    print('average path length:', np.mean(average_distance))
    

def main(datafile, model_path):
    data, net = load_model(datafile, model_path)
    print(datafile)
    print(model_path)
    test(data, net)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', '-d', type=str, default='dataset/test_worlds/sparse_16_20_2000.npy')
    parser.add_argument('--model_path', '-m', type=str, default='saved_models/2024-09-21-14-29-57_FINAL_10x10_32_x200.pt')
    args = parser.parse_args()
    main(args.datafile, args.model_path)