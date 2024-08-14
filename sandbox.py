import numpy as np
import matplotlib.pyplot as plt
import torch

from generators.sparse_map import  SparseMap
from model import VIN


# generate ONE world
num_envs = 1
map_side_len = 16
obstacle_percent = 20
scale = 2

env = SparseMap.genMaps(num_envs, map_side_len, obstacle_percent, scale)[0] # 0 = 0 freespace, 1 = obstacle
world = env.genPOMDP() # equivalent to what "gridworld" was, stores all info like T, O, R, etc.
    # i sort of don't like how this is implemented ? only 4 actions, not "in order"
    # indexing is [a, s, s'] instead of [s, a, s']
    # gridworld seems fancier with how it gets the transitions but this also makes more sense maybe...
    # also could remove stuff like "addPath" etc because we don't need expert solvers

    # twoD_map_path is more focused on representing the expert paths and actions that I don't need for my implementation
    # is it worth it to rewrite/reorganize the code so the "world" class is just basic map info/functions, gridworld/pomdp representations, and image view?
    

# visualize the world
plt.ion()
fig, ax = plt.subplots()
plt.imshow(env.grid.T, cmap="Greys")
ax.plot(env.goal_r, env.goal_c, 'bd')

action_mapping = ["right", "up", "left", "down", "stay"]
# start somewhere and move.
start_state = np.random.randint(len(world.states)) # random start state idx

### if u loop this --> line 42 it will be a random walk simulation 
action = np.random.choice(world.n_actions) # random action idx
next_state = np.random.choice(range(world.n_states), p=world.T[action, start_state]) # next state based on action and current state
observation = np.random.choice(range(len(world.observations)), p=world.O[action, next_state]) # um what are these observations/what do they mean...
reward = world.R[action, start_state, next_state, observation] 

#visualize that ?
state_x, state_y = SparseMap.roomIndexToRc(env.grid, start_state)
ax.plot(state_x, state_y, 'ro')
state_x, state_y = SparseMap.roomIndexToRc(env.grid, next_state)
ax.plot(state_x, state_y, 'go')
print(action, action_mapping[action])
start_state = next_state

# end at goal
goal_state = SparseMap.rcToRoomIndex(env.grid, env.goal_r, env.goal_c)
if start_state == goal_state:
    print("Reached goal!")


# store the state movements i guess
# as a dict? objedct? need to be able to shufflr so list of dicts for now
experiences = []

# how to distinguish between seperate trajectories
# how to shuffle and train

config = {
    "imsize": map_side_len + 2, 
    "lr": 0.005,
    'epochs': 30,
    'l_i': 2,
    'l_h': 150,
    "l_q": 10,
    "k": 20,
}

# create "input_view", which is just VIN representation for the reward... ?
# would be nice to store this within the world object
reward_mapping = -1 * np.ones(env.grid.shape) # -1 for freespace
reward_mapping[env.goal_r, env.goal_c] = 10 # 10 at goal
grid_view = np.reshape(env.grid, (1, 1, env.grid.shape[0], env.grid.shape[1]))
reward_view = np.reshape(reward_mapping, (1, 1, env.grid.shape[0], env.grid.shape[1]))
input_view = torch.Tensor(np.concatenate((grid_view, reward_view), axis=1)) # inlc empty 1 dim


model = VIN(config)
r, v = model.process_input(input_view)
for i in range(config['k'] - 1):
    q = model.eval_q(r, v)
    v, _ = torch.max(q, dim=1, keepdim=True)

q = model.eval_q(r, v) 


##### CODE AFTER THIS IS NOT DONE YET 
state_x, state_y = experience['current_state'] 
q_pred = q[0, :, state_x, state_y] # pull state_x, state_y from experiences['current_state']
q_target = experience['reward'] # pull experiences from stored actions


# do something like this.... now just have to reorganize how the expriences are stored....
if not experience['done']:
    q_target = q_target + self.gamma * np.max(self.model.q_values[0, :, experience['next_state'][0], experience['next_state'][1]].detach().numpy())
q_current = self.model.fc(q_current) # if map to actions, should i do this to all q_vals or just q_current?
q_current[experience['action']] = q_target # hm this only works with q_current having len = n_actions...
# self.model.fit(experience['current_state'], q_current, verbose=0)
pred, output = self.model.fc(self.model.q_values[0, :, state_x, state_y]), q_current
loss = criterion(pred, output)
loss.backward()
optimizer.step()
