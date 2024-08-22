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

    def forward(self, input_view, state_x, state_y, k):
        """
        :param input_view: (batch_sz, imsize, imsize)
        :param state_x: (batch_sz,), 0 <= state_x < imsize
        :param state_y: (batch_sz,), 0 <= state_y < imsize
        :param k: number of iterations
        :return: logits and **argmax** of logits
        """
        r, v = self.process_input(input_view)

        # Update q and v values
        q = self.value_iteration(r, v)
        
        logits, action = self.get_action(q, state_x, state_y)

        return q, logits, action
    
    def process_input(self, input_view):
        h = self.h(input_view)  # Intermediate output
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