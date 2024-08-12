import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
from collections import OrderedDict
from itertools import product

class Worlds:
    # basically the gridworld class but it can take in batches as input i guess.
    
    ACTION = OrderedDict(N=(-1, 0), S=(1, 0), E=(0, 1), W=(0, -1), NE=(-1, 1), NW=(-1, -1), SE=(1, 1), SW=(1, -1))

    def __init__(self, image, target_x, target_y):
        self.image = image # false/0 = freespace, true/1 = obstacle.
        self.n_worlds = image.shape[0]
        self.n_row = image.shape[1]
        self.n_col = image.shape[2]
        self.obstacles = [np.where(self.image[i] == 0) for i in range(self.n_worlds)] # world, row, col where there are obstacles? i hope.
        self.freespace = [np.where(self.image[i] != 0) for i in range(self.n_worlds)]
        self.target_x = target_x
        self.target_y = target_y
        self.n_states = self.n_row * self.n_col
        self.n_actions = len(self.ACTION)

        self.G, self.W, self.P, self.R, self.iv_mixed = self.set_vals()
        self.start_x, self.start_y = self.gen_start() # if i use koosha's method of generating graph I dont actually need this bc its already fully connected ? i can just generate a random freespace

    def loc_to_state(self, row, col):
        return [np.ravel_multi_index([row[i], col[i]], (self.n_row, self.n_col), order='F') for i in range(len(row))]
        # not sure if this can be done more efficently 
        # i used this initially but its not as compatible later if we wanna index the rows
        # return np.ravel_multi_index([range(self.n_worlds), row, col], (self.n_worlds, self.n_row, self.n_col)) # also would have to take out order='F'
    
    def get_coords(self, state):
        return [np.unravel_index(state, (self.n_row, self.n_col), order='F') for i in range(len(state))]
    
    def move(self, world, row, col, action):
        r_move, c_move = self.ACTION[action]
        new_row = [np.maximum(0, np.minimum(row[i] + r_move, np.repeat(self.n_row - 1, self.n_row))) for i in range(self.n_row)]
        new_col = [np.maximum(0, np.minimum(col[i] + c_move, np.repeat(self.n_col - 1, self.n_col))) for i in range(self.n_col)]
        new_row, new_col = np.where(self.image[world, new_row, new_col] == 0, (row, col), (new_row, new_col)) # if bump into obstace, stay in place
        return new_row, new_col

    def set_vals(self):
        # Setup function to initialize all necessary

        # Cost of each action, equivalent to the length of each vector
        #  i.e. [1., 1., 1., 1., 1.414, 1.414, 1.414, 1.414]
        action_cost = np.linalg.norm(list(self.ACTION.values()), axis=1)
        
        # determine rewards
        # R = self.get_reward_prior().ravel('F')

        R = - np.ones((self.n_worlds, self.n_states, self.n_actions)) * action_cost
        # Reward at target is --zero-- *ten*
        target = self.loc_to_state(self.target_x, self.target_y)
        R[range(self.n_worlds), target, :] = 10 # works less well if reward = 0, still works sometimes?

        # Transition function P: (curr_state, next_state, action) -> probability: float
        P = np.zeros((self.n_worlds, self.n_states, self.n_states, self.n_actions))
        # Filling in P
        for world in range(self.n_worlds):
            for i_action, action in enumerate(self.ACTION):
                row = np.array([[i for i in range(self.n_col)] for _ in range(self.n_row)])
                col = np.array([[i for _ in range(self.n_col)] for i in range(self.n_row)])
                # row = [i for i in range(self.n_row) for _ in range(self.n_col)] # this i cur [1:18, [1:18], needs to be [1*18 : 18*18], [1:18 * 18]]///
                # col = [i for _ in range(self.n_row) for i in range(self.n_col)] # 1: does this works, 2: is it efficent
                neighbor_row, neighbor_col = self.move(world, row, col, action)
                neighbor_state = self.loc_to_state(neighbor_row.flatten(), neighbor_col.flatten())
                P[world, range(self.n_states), neighbor_state, i_action] = 1


        # Adjacency matrix of a graph connecting curr_state and next_state
        G = np.logical_or.reduce(P, axis=3) # i just changed axis from 2 to 3 to account for extra dim??
        # Weight of transition edges, equivalent to the cost of transition
        W = np.maximum.reduce(P * action_cost, axis=3)

        # If inputting multiple gridworlds, there are different amounts of non-obstacles...
        # yielding a non-homogenous array which can't be converted into np :\
        # solution ?: skip the non_obstacles part...

        iv_mixed = np.concatenate((self.image.reshape(self.n_worlds, 1, self.n_row, self.n_col), self.get_reward_prior().reshape(self.n_worlds, 1, self.n_row, self.n_col)), axis=1)
        return G, W, P, R, iv_mixed 
    
    def get_reward_prior(self): # -1 all, 10 targrt....
        im = -1 * np.ones((self.n_worlds, self.n_row, self.n_col))
        im[range(self.n_worlds), self.target_x, self.target_y] = 10
        return im

    def gen_start(self):
        g_dense = np.transpose(self.W, (0, 2, 1)) # uhhh does this still work when its no longer sparse/freespace representation ?
        g_sparse = [csr_matrix(dense) for dense in g_dense] # is it worth to convert to sparse matrix for dijkstra? 
        goal_s = self.loc_to_state(self, self.target_x, self.target_y)
        pred = np.array([dijkstra(g_sparse[i], indices=goal_s[i], return_predecessors=True)[1] for i in range(self.n_worlds)])
        # pred = pred[:, 1]
        cc = np.array([connected_components(g_sparse[i], directed=False, return_labels=True)[1] for i in range(self.n_worlds)])
        cc_idx = [np.where(cc[i] == cc[i, goal_s[i]]) for i in range(self.n_worlds)]
        start_x, start_y = self.get_coords(self, np.random.choice(cc_idx[1]))

        #uhhh i think the calculation of goal.
        # or the calculation of goal_s 
        # is wrong ?

    
    def __size__(self):
        return self.image.shape[0]

import torch
import torch.utils.data as data

class WorldDataset(data.Dataset):
    def __init__(self, file, imsize, train=True):
        assert file.endswith('.npz')
        self.file = file
        self.imsize = imsize
        self.train = train

        self.images = self._process(file, self.train)
    
    def _process(self, file, train):
        with np.load(file, mmap_mode='r', allow_pickle=True) as f:
            images = f['arr_0']
            goals = f['arr_1']
        # set proper datatypes in og code - do i need that?
        if train:
            print("Number of Train Samples: {0}".format(images.shape[0]))
        else:
            print("Number of Test Samples: {0}".format(images.shape[0]))
        return images, goals
    def __getitem__(self, index):
        images = self.images[index]
        goals = self.goals[index]
        torch.from_numpy(images)
        return images, goals
    def __len__(self):
        return self.images.shape[0]

class DatasetFromArray(data.Dataset):
    def __init__(self, images, goals, train=True):
        self.images = images
        self.goals = goals
        self.train = train
    def __getitem__(self, index):
        img = self.images[index]
        goal = self.goals[index]
        torch.from_numpy(img) # uh what this fo
        return img, goal
    def __len__(self):
        return self.images.shape[0]
    
    