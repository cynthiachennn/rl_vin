from dataset.dataset import *
from dataset.make_training_data import *
from domains.gridworld import *
from generators.obstacle_gen import *
from model import *

np.random.seed(9)

### functions that make up the underlying code
# first generate data
def generate_gridworld(max_obs, dom_size):  # "random" gridworld
    goal = [np.random.randint(dom_size[0]), np.random.randint(dom_size[1])]
    obs = obstacles([8, 8], goal, max_obs) # can use obs.show() to show obs map - black = free, white = obstacle...
    n_obs = obs.add_n_rand_obs(max_obs)
    border_res = obs.add_border()
    im = obs.get_final()
    G = GridWorld(im, goal[0], goal[1])
    return G

def make_gridworld(): # predeterminate gridworld
    goal = [3, 0]
    img = [[0 for i in range(8)],
           [0, 1, 0, 0, 0, 0, 1, 0],
           [0, 1, 1, 1, 1, 1, 1, 0],
           [0, 1, 0, 0, 0, 0, 1, 0],
           [0, 1, 1, 1, 1, 1, 1, 0], 
           [0, 1, 1, 0, 0, 0, 1, 0],
           [0, 1, 0, 1, 1, 1, 1, 0],
           [0 for i in range(8)]]
    G = GridWorld(img, goal[0], goal[1])
    return G

def get_sample(G, n_traj, i, dom_size):
    value_prior = G.t_get_reward_prior()
    states_xy, states_one_hot = sample_trajectory(G, n_traj)
    actions = extract_action(states_xy[i])
    ns = states_xy[i].shape[0] - 1# Invert domain image => 0 = free, 1 = obstacle
    image = 1 - G.image
    image_data = np.resize(image, (1, 1, dom_size[0], dom_size[1]))
    value_data = np.resize(value_prior,
                            (1, 1, dom_size[0], dom_size[1]))
    iv_mixed = np.concatenate((image_data, value_data), axis=1)
    X_current = np.tile(iv_mixed, (ns, 1, 1, 1)) # img/val * ns (so theres a seperate representation for each starting state generated)
    S1_current = np.expand_dims(states_xy[i][0:ns, 0], axis=1)
    S2_current = np.expand_dims(states_xy[i][0:ns, 1], axis=1)
    Labels_current = np.expand_dims(actions, axis=1)
    # optional: show the graph b4 continuing...
    def visualize(G, states_xy):
        plt.ion()
        fig, ax = plt.subplots()
        implot = plt.imshow(G.image.T, cmap="Greys_r") # why the .T again....
        ax.plot(states_xy[:, 0], states_xy[:, 1], c='b', label='Optimal Path')
        ax.plot(states_xy[0, 0], states_xy[0, 1], 'ro', label='Start')
        ax.plot(states_xy[-1, 0], states_xy[-1, 1], 'go', label='Goal')
    
    visualize(G, states_xy[i])
    return X_current, S1_current, S2_current, Labels_current

def visualize(G, start=None, goal=None): # ugh sorta redundant but not worth cleaning up
    plt.ion()
    fig, ax = plt.subplots()
    implot = plt.imshow(G.image.T, cmap='Greys_r') # WHY T
    if start is not None:
        ax.plot(start[0], start[1], 'ro', label='Start')
    if goal is not None:
        ax.plot(goal[0], goal[1], 'go', label='Goal')


# then create nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter

def create_nn():
    vin = dict()
    vin['h'] = nn.Conv2d(
        in_channels=2,
        out_channels=150,
        kernel_size=(3, 3),
        stride=1,
        padding=1,
        bias=True)
    vin['r'] = nn.Conv2d(
        in_channels=150,
        out_channels=1,
        kernel_size=(1, 1),
        stride=1,
        padding=0,
        bias=False)
    vin['q_func'] = nn.Conv2d(
        in_channels=1,
        out_channels=10,
        kernel_size=(3, 3),
        stride=1,
        padding=1,
        bias=False)
    vin['fc'] = nn.Linear(in_features=10, out_features=8, bias=False)
    vin['w'] = Parameter(
        torch.zeros(10, 1, 3, 3), requires_grad=True)
    vin['sm'] = nn.Softmax(dim=1)
    return vin

def forward(X_current, S1_current, S2_current, vin):
    X_current = torch.tensor(X_current, dtype=torch.float32)
    S1_current = torch.tensor(S1_current.flatten(), dtype=torch.float32)
    S2_current = torch.tensor(S2_current.flatten(), dtype=torch.float32)
    vin['h'] = vin['h'](X_current)
    vin['r'] = vin['r'](vin['h'])
    vin['q'] = vin['q_func'](vin['r'])
    vin['v'], _ = torch.max(vin['q'], dim=1, keepdim=True)
    def eval_q(r, v):
        return F.conv2d(
            # Stack reward with most recent value
            torch.cat([r, v], 1),
            # Convolve r->q weights to r, and v->q weights for v. These represent transition probabilities
            torch.cat([vin['q_func'].weight, vin['w']], 1),
            stride=1,
            padding=1)
    # Update q and v values
    k = 100
    for i in range(k - 1):
        vin['q'] = eval_q(vin['r'], vin['v'])
        vin['v'], _ = torch.max(vin['q'], dim=1, keepdim=True)
    vin['q'] = eval_q(vin['r'], vin['v'])
    # q: (batch_sz, l_q, map_size, map_size)
    batch_sz, l_q, _, _ = vin['q'].size()
    q_out = vin['q'][torch.arange(batch_sz), :, S1_current.long(), S2_current.long()].view(batch_sz, l_q)
    pred_action = vin['fc'](q_out)  # q_out to actions
    return pred_action, vin['sm'](pred_action)

class Agent():
    def __init__(self, config):
        self.config = config
        self.model = VIN(config)
        self.exploration_prob = 1.0
        self.n_actions = 8 # this shouldn't change? only a variable bc example is in continuous state space
        self.memory_buffer = []
        self.max_memory_buffer = 2000
        self.gamma = 0.99

    def compute_action(self, current_state):
        if np.random.uniform(0, 1) < self.exploration_prob:
            print('random')
            return np.random.choice(range(self.n_actions))
        # q_values = self.model.forward(self.state)
        print('policy')
        q_values = self.model.map_qs(self.iv_mixed, self.config.k)
        # action = np.max(self.model.fc(q_values[0, :, current_state[0], current_state[1]]))
        _, action = self.model.forward(self.iv_mixed, current_state[0], current_state[1], self.config.k)
        return np.argmax(action.detach().numpy())

    def update_exploration_prob(self, decay = 0.005):
        self.exploration_prob = self.exploration_prob * np.exp(-decay)
        # print(f'Exploration probability: {self.exploration_prob}')
    
    def store_episode(self, current_state, action, reward, next_state, done):
        self.memory_buffer.append({'current_state': current_state,
                                    'action': action,
                                    'reward': reward,
                                    'next_state': next_state,
                                    'done': done})
        if len(self.memory_buffer) > self.max_memory_buffer: # i guess to avoid info overload? copied this from tds guide
            self.memory_buffer.pop(0)
    
    def learn_world(self, G):
        # Invert domain image => 0 = free, 1 = obstacle
        image = 1 - G.image
        image_data = np.resize(image, (1, 1, self.config.imsize, self.config.imsize))
        value_data = np.resize(G.t_get_reward_prior(),
                                (1, 1, self.config.imsize, self.config.imsize))
        self.iv_mixed = torch.Tensor(np.concatenate((image_data, value_data), axis=1))
        self.q_values = self.model.map_qs(self.iv_mixed, self.config.k)
        # q = self.model.fc(q) # idk if we need to map to actions...

    def train(self, batch_size, criterion, optimizer):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # idk just adding this cuz
        # sample from memory buffer
        batch = np.random.permutation(self.memory_buffer)
        batch = batch[:batch_size]

        for experience in batch: # experience is a singular action
            # problem: i'm doing this per experience so not taking advantage of being able to use
            # multiple batches ? should put multiple experiences in a batch to train at once
            # would have to update the pred, output line thoo cuz currently operates on model.fc 
            # on one batch at a time.
            self.model = self.model.to(device)
            optimizer.zero_grad()
            state_x, state_y = experience['current_state']
            q_values = self.model.map_qs(self.iv_mixed, self.config.k) # equivalent to "self.forward?"
            # ^do i need this if i already have self.q_values from learn_world? how often
            # should i update the q_values
            q_current = self.model.q_values[0, :, state_x, state_y] # hm what does the model know...
                                                # cuz i feel like VIN formula needs the model to know
                                                # the map/observation not just state...
                                                # but i wanna keep the agent seperate from the env...
                                                # cuz it shouldn't know....
            
            q_target = experience['reward']
            if not experience['done']: # hm is this okay to implement explicity like this yes i think so 
                q_target = q_target + self.gamma * np.max(self.model.q_values[0, :, experience['next_state'][0], experience['next_state'][1]].detach().numpy())
            q_current = self.model.fc(q_current) # if map to actions, should i do this to all q_vals or just q_current?
            q_current[experience['action']] = q_target # hm this only works with q_current having len = n_actions...
            # self.model.fit(experience['current_state'], q_current, verbose=0)
            pred, output = self.model.fc(self.model.q_values[0, :, state_x, state_y]), q_current
            loss = criterion(pred, output)
            loss.backward()
            optimizer.step()
            

    def train(self, batch_size, criterion, optimizer, q_target):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # idk just adding this cuz
        # sample from memory buffer
        batch = np.random.permutation(self.memory_buffer)
        batch = batch[:batch_size]
        q_target = np.zeros((self.config.imsize, self.config.imsize, 8))

        for experience in batch: # experience is a singular action
            # problem: i'm doing this per experience so not taking advantage of being able to use
            # multiple batches ? should put multiple experiences in a batch to train at once
            # would have to update the pred, output line thoo cuz currently operates on model.fc 
            # on one batch at a time.
            self.model = self.model.to(device)
            optimizer.zero_grad()
            state_x, state_y = experience['current_state']
            q_values = self.model.map_qs(self.iv_mixed, self.config.k) # equivalent to "self.forward?"
            q_target = experience['reward'] + self.gamma * np.max(q_target[experience['next_state']])
            # ^do i need this if i already have self.q_values from learn_world? how often
            # should i update the q_values
            pred, output = self.model.fc(q_values[0, :, state_x, state_y]), torch.tensor(q_target[experience['current_state']])
            print(pred.shape)
            
            loss = criterion(pred, output)
            loss.backward()
            optimizer.step()     
            
def move(G, agent, current_state, action): # this is so we can input a guarenteed traj lol 
    next_state = G.sample_next_state(current_state, action)
    G.state_to_loc(next_state)
    reward = G.R[next_state] # is it bad to directly access... should i make a getter?
    if next_state == G.map_ind_to_state(G.target_x, G.target_y):
        done = True
    # agent.store_episode(current_state, action, reward, next_state, done)
    agent.memory_buffer.append({'current_state': current_state,
                                        'action': action,
                                        'reward': reward,
                                        'next_state': next_state,
                                        'done': done})
    if done: agent.update_exploration_prob() # add break if done so the agent stops moving
    # then restart cycle/search from beginning
    current_state = next_state


            
         
