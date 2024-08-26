import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter


class VIN(nn.Module):
    def __init__(self, config):
        super(VIN, self).__init__()
        self.config = config
        self.h = nn.Conv2d(
            in_channels=config['l_i'],
            out_channels=config['l_h'],
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True)
        self.r = nn.Conv2d(
            in_channels=config['l_h'],
            out_channels=1,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=False)
        self.q = nn.Conv2d(
            in_channels=1,
            out_channels=config['l_q'],
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=False)
        self.fc = nn.Linear(in_features=config['l_q'], out_features=config['n_act'], bias=False)
        self.w = Parameter(
            torch.zeros(config['l_q'], 1, 3, 3), requires_grad=True)
        self.sm = nn.Softmax(dim=1)

        self.exploration_prob = 1 # start with only exploring
        self.rng = np.random.default_rng() # should i pass this in... ? ..
        self.n_act = config['n_act']
        self.device = config['device']

    def forward(self, world, current_state, max_steps, test=False):
        """
        :param input_view: (batch_sz, imsize, imsize)
        :param state_x: (batch_sz,), 0 <= state_x < imsize
        :param state_y: (batch_sz,), 0 <= state_y < imsize
        :param k: number of iterations
        :return: logits and **argmax** of logits
        """
        trajectory = np.empty((0, 5), dtype=int)
        total_steps = 0
        done = 0
        inputView = torch.Tensor(world.inputView)
        r, v = self.process_input(inputView)
        q = self.value_iteration(r, v)
        while done == 0 and total_steps < max_steps:
            state_x, state_y = world.roomIndexToRc(current_state)
            if test == False:
                if self.rng.random() < self.exploration_prob:
                    action = self.rng.choice(self.n_act) # hh storing everything in self.
                else:
                    logits, action = self.get_action(q, state_x, state_y)
            else: # test = true means always follow policy
                logits, action = self.get_action(q, state_x, state_y)
            next_state, reward, done = self.move(world, action, state_x, state_y)
            trajectory = np.vstack((trajectory, [current_state, action, reward, next_state, done]))
            current_state = next_state
            total_steps += 1
        
        trajectory = trajectory.astype(int)

        if test == False:
            q_target = [np.sum(trajectory[i:, 2]) for i in range(trajectory.shape[0])]
            pred_values = torch.empty((0))
            for episode in trajectory:
                state_x, state_y = world.roomIndexToRc(episode[0])
                q_pred, _ = self.get_action(q, state_x, state_y)
                pred_values = torch.cat((pred_values, q_pred))
            target_values = torch.clone(pred_values)
            target_values[:, trajectory[:, 1]] = torch.Tensor(q_target).to(self.device)
            
            return torch.stack((pred_values, target_values))
        
        if test == True:
            return trajectory
    
    def process_input(self, inputView):
        h = self.h(inputView)  # Intermediate output
        r = self.r(h)           # Reward
        q = self.q(r)           # Initial Q value from reward
        v, _ = torch.max(q, dim=1, keepdim=True)

        return r, v

    def value_iteration(self, r, v):
        for i in range(self.config['k'] - 1):
            q = self.eval_q(r, v)
            v, _ = torch.max(q, dim=1, keepdim=True)
        q = self.eval_q(r, v)
        return q
    
    def eval_q(self, r, v):
        return F.conv2d(
            # Stack reward with most recent value
            torch.cat([r, v], 1),
            # Convolve r->q weights to r, and v->q weights for v. These represent transition probabilities
            torch.cat([self.q.weight, self.w], 1),
            stride=1,
            padding=1)

    def get_action(self, q, state_x, state_y):
        batch_sz, l_q, _, _ = q.size()
        # q_out = q[torch.arange(batch_sz), :, state_x.long(), state_y.long()].view(batch_sz, l_q)
        q_out = q[torch.arange(batch_sz), :, state_x, state_y].view(batch_sz, l_q)
        logits = self.fc(q_out)  # q_out to actions
        action = torch.argmax(logits) # orignal action has nn.softmax, which i could np.choice with p=action to choose from distribution instead ? 
        return logits, action

    def move(self, world, action, state_x, state_y):
        current_state = world.rcToRoomIndex(state_x, state_y)
        next_state = self.rng.choice(range(len(world.states)), p=world.T[action, current_state])
        # observation = self.rng.choice(range(len(world.observations)), p=world.O[action, next_state])
        reward = world.R[action, current_state, next_state, 0] # 0 for now since they're all the same, but should be based on observation
        if state_x == world.goal_r and state_y == world.goal_c:
            done = 1
        else:
            done = 0
        
        return next_state, reward, done