from tracemalloc import start
from typing_extensions import Final
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime
# from tqdm import tqdm
import argparse

from model import VIN
from test import test
from train import train

rng = np.random.default_rng(9) # is this bad to do?

import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

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
        "lr": 0.005,
        'l_i': 2,
        'l_h': 150,
        "l_q": 10,
        "k": 20,
        "max_steps": 50,
    }

    net = VIN(config).to(device)
    net = torch.nn.DataParallel(net)

    net.load_state_dict(torch.load('saved_models/2024-09-10-18-20-26_FINAL_6x6_32_x400.pt'))

    train(worlds, net, config, epochs, batch_size)
    test(worlds_test, net, viz=False)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainfile', '-train', default='dataset/train_worlds/sparse_8_20_20000.npy')
    parser.add_argument('--testfile', '-test', default='dataset/test_worlds/sparse_8_20_2000.npy')
    parser.add_argument('--epochs', '-e', default=200)
    parser.add_argument('--batch_size', '-b', default=32)
    args = parser.parse_args()
    
    main(args.trainfile, args.testfile, args.epochs, args.batch_size)    
