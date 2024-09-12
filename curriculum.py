import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from tqdm import tqdm
import argparse

from model import VIN
from test import test
from train import train

rng = np.random.default_rng(9) # is this bad to do?

import os
os.environ["CUDA_VISIBLE_DEVICES"]= "2"

if __name__ == '__main__':
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
        "max_steps": 50,
    }

    net = VIN(config).to(device)
    net = torch.nn.DataParallel(net)

    trainfiles = ['dataset/train_worlds/small_4_4_20000.npy', 'dataset/train_worlds/sparse_16_20_20000.npy']
    epochs = 200
    batch_size = 32
    for file in trainfiles:
        worlds = np.load(file)
        imsize = worlds[0].shape[0]
        config['max_steps'] = 3 * imsize
        train(worlds, net, config, epochs, batch_size)
    

    testfile = 'dataset/test_worlds/sparse_16_20_2000.npy'
    worlds_test = np.load(testfile)
    test(worlds_test, net, viz=False)