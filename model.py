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
        self.exploration_prob = 1
        self.actions = [[0, 1], [-1, 0], [0, -1], [1,0], [0, 0]] # ["right", "up", "left", "down", "stay"]
        self.rng = np.random.default_rng(9) ## < idk
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

        self.fc = nn.Linear(in_features=config['l_q'], out_features=len(self.actions), bias=False)
        self.w = Parameter(
            torch.zeros(config['l_q'], 1, 3, 3), requires_grad=True)
        self.sm = nn.Softmax(dim=1)

    def forward(self, input_view, coords, mode='rl'):
        # maybe track the evolution of q_value
        state_x, state_y, goal_x, goal_y = torch.transpose(coords, 0, 1)
        device = self.config['device'] #input_view.get_device()   ### module
        batch_size = input_view.shape[0]
        trajectory = torch.empty(size=(0, batch_size, 5), dtype=torch.int, device=device)
        logitsList = torch.empty(size=(0, batch_size, len(self.actions)), device=device)
        total_steps = 0
        done = torch.ones(batch_size, device=device)
        r, v = self.process_input(input_view)
        q = self.value_iteration(r, v)
        while torch.any(done == 1) and total_steps < self.config['max_steps']:
            if mode == 'rl': # only explore if training rl
                if self.rng.random() < self.exploration_prob:
                    action = torch.tensor(self.rng.choice(len(self.actions), batch_size), device=device)
                else:
                    logits, action = self.get_action(q, state_x, state_y)
            else:
                logits, action = self.get_action(q, state_x, state_y)
            next_x, next_y, reward = self.move(input_view, action, state_x, state_y)
            # done = 0 means goal has been reached
            done = done * torch.where(done == 2, 0, 1)
            done = done * torch.where(((next_x == goal_x) & (next_y == goal_y)), 2, 1)
            experience = torch.vstack((state_x, state_y, action, reward, done)).transpose(0, 1).reshape((1, batch_size, 5))
            trajectory = torch.cat((trajectory, experience))
            state_x, state_y = next_x, next_y
            if mode == 'expert':
                logitsList = torch.cat((logitsList, logits.unsqueeze(0)))
            total_steps += 1
        trajectory = trajectory.to(torch.int)

        if mode == 'rl':
            trajectory[:, :, 3] = trajectory[:, :, 3] * trajectory[:, :, 4] # set reward past done to 0
            q_target = [torch.sum(trajectory[i:, range(batch_size), 3], dim=0) * 0.2 for i in range(trajectory.shape[0])]
            q_target = torch.stack(q_target).to(device).to(torch.float) # q_target.shape = [n_episodes, batch_size]
            pred_values = torch.empty((0, batch_size, len(self.actions)), device=device) # [episodes, batch_size, n_actions]
            for episode in trajectory:
                state_x, state_y = episode[range(batch_size), 0], episode[range(batch_size), 1]
                q_pred, _ = self.get_action(q, state_x, state_y)
                pred_values = torch.cat((pred_values, q_pred.unsqueeze(0)))
            target_values = torch.clone(pred_values)
            # get target values based on the state in that episode
            indices = trajectory[:, :, 2]
            batch_indices = torch.arange(target_values.size(0)).view(-1, 1).expand_as(indices)
            column_indices = torch.arange(target_values.size(1)).expand_as(indices)
            target_values[batch_indices, column_indices, indices] = q_target
            pred_values = torch.einsum('eba,eb->eba', [pred_values, trajectory[:, :, 4]]) # for ones that are not past Done, set predicted value... ?
            target_values = torch.einsum('eba,eb->eba', [target_values, trajectory[:, :, 4]]) 
            return torch.stack((pred_values, target_values))
        if mode == 'test':
            return trajectory
        if mode == 'expert':
            return logitsList, trajectory
    
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
        action = torch.argmax(logits, dim=1) # orignal action has nn.softmax, which i could np.choice with p=action to choose from distribution instead ? 
        return logits, action

    def move(self, input_view, actions, state_x, state_y):
        device = self.config['device'] # input_view.get_device() ## this is so redundant ahh #module
        grid, reward = input_view[:, 0], input_view[:, 1]
        action = [self.actions[action] for action in actions]
        action = torch.tensor(action, dtype=torch.int, device=device)
        next_x, next_y = state_x + action[:, 0], state_y + action[:, 1]
        for i in torch.where(grid[range(input_view.shape[0]), next_x, next_y] == 1):
            next_x[i] = state_x[i]
            next_y[i] = state_y[i]
        reward = reward[range(input_view.shape[0]), next_x, next_y]
        return next_x, next_y, reward