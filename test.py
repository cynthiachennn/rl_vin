import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse

from model import VIN
from domains.Worlds import World

rng = np.random.default_rng()

def load_model(datafile, model_path):
        
    datafile = 'dataset/saved_worlds/small_4_4_64.npy' # type_size_density_n_envs

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
    }

    model_path = 'saved_models/2024-08-28 00-05-13_6x6_51_x32.pt'

    model = VIN(config).to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return worlds, model

def test(data, model):
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

            # create input view
            reward_mapping = -1 * np.ones(world.shape) # -1 for freespace
            reward_mapping[goal_x, goal_y] = 20 # what value at goal? also if regenerating goals for each world then i'd need to redo this for each goal/trajectory.
            world[goal_x, goal_y] = 0 # remove goal from grid view
            grid_view = np.reshape(world, (1, 1, world.shape[0], world.shape[1]))
            reward_view =  np.reshape(reward_mapping, (1, 1, world.shape[0], world.shape[1]))
            input_view = np.concatenate((grid_view, reward_view), axis=1) # inlc empty 1 dim
            input_view = torch.Tensor(input_view)

            trajectory = model(input_view, start_x, start_y, world.shape[0]*world.shape[1], test=True) # max steps = size of world?
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
                q_max = [[np.max(model.module.get_action(q, r, c)[0].cpu().detach().numpy()) for c in range(world.shape[1])] for r in range(world.shape[0])]
                plt.imshow(q_max, cmap='viridis')
                plt.show()
                
            if trajectory[-1, 4] ==False:
                print('failed?')
    print('accuracy:', correct/len(data))

def main(datafile, model_path):
    data, model = load_model(datafile, model_path)
    test(data, model)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    datafile = parser.add_argument('--datafile', '-d', type=str, default='dataset/saved_worlds/small_4_4_64.npy')
    model_path = parser.add_argument('--model_path', '-m', type=str, default='saved_models/2024-08-28 00-05-13_6x6_51_x32.pt')
    main(datafile, model_path)