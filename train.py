import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from tqdm import tqdm

from dataset.generate_rl_dataset import SmallMap
from model import VIN
from domains.batch_worlds import World

def load_data(datafile):
    with np.load(datafile) as f:
        envs = f['arr_0']
        goals = f['arr_1']
    n_envs = envs.shape[0]
    imsize = envs.shape[1]    
    worlds = [World(envs[i], goals[i][0], goals[i][1]) for i in range(len(envs))]
    rng.shuffle(worlds)
    worlds_train = worlds[:int(0.8 * len(worlds))]
    worlds_test = worlds[int(0.8 * len(worlds)):]
    return worlds_train, worlds_test


def main(datafile, epochs, batch_size):
    worlds_train, worlds_test = load_data(datafile)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', '-df', default='dataset/rl/small_4_4_1024.npz')
    parser.add_argument('--epochs', '-e', default=32)
    parser.add_argument('--batch_size', '-b', default=32)
    args = parser.parse_args()
    
    main(args.datafile, args.epochs, args.batch_size)    
