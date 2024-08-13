import matplotlib.pyplot as plt
from utility.utils import *
from sandbox_utils import *

# WIP
from generators.sparse_map import SparseMap
from domains.batch_worlds import *


# generate data?? -sparse map not kentsommer gridworld.
np.random.seed(20)
num_envs = 16
batch_size = 4
map_side_len = 16
obstacle_percent = 20
scale = 2

all_envs = SparseMap.genMaps(num_envs, map_side_len, obstacle_percent, scale)
images = np.array([env.grid for env in all_envs])
images = np.logical_not(images)
goals = np.array([(env.goal_r, env.goal_c) for env in all_envs])
trainset = DatasetFromArray(images, goals, train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

image, goal = trainset.images[0:4], trainset.goals[0:4]
worlds = Worlds(image, goal[:, 0], goal[:, 1])

i = 0
plt.ion() 
fig, ax = plt.subplots()
plt.xlabel('X/RoWS')  # Add X axis label
plt.ylabel('Y/Col')  # Add Y axis label
plt.imshow(worlds.image[i].T, cmap="Greys_r", alpha=0.7)
# Overlay numbers on the gridworld
for row in range(worlds.n_row):
    for col in range(worlds.n_col):
        number = row * worlds.n_col + col  # Calculate the number for each cell
        # number = int(worlds.loc_to_state(np.array([row]), np.array([col]))[0])
        # ^ THIS IS WRONG !!!
        # number = np.ravel_multi_index([row, col], (worlds.n_row, worlds.n_col))
        text = ax.text(col, row, str(number), fontsize=9, color='red', ha='center', va='center')
ax.plot(worlds.target_x[i], worlds.target_y[i], 'bo')

# model one transition function
curr_state = 0
action = 3 #East
state_x, state_y = worlds.get_coords(curr_state)
ax.plot(state_x, state_y, 'ro')

rewards = np.sum(worlds.R[i], axis=1).reshape((worlds.n_row, worlds.n_col))
plt.imshow(rewards, cmap="Reds", alpha=0.3)
show_transitions = worlds.P[i, curr_state, :, action].reshape((worlds.n_row, worlds.n_col))

from matplotlib.colors import LinearSegmentedColormap
colors = [(1, 1, 1, 0), (0, 0, 1)]  # From transparent to blue
cm = LinearSegmentedColormap.from_list('blue/clear', colors, N=100)
# Now use this colormap in your imshow call
plt.imshow(show_transitions, cmap=cm, alpha=0.2)
plt.imshow(worlds.G[i, curr_state, :].reshape((worlds.n_row, worlds.n_col)), cmap="Greens", alpha=0.6)

# model connected components?
g_dense = np.transpose(worlds.W, (0, 2, 1)) # uhhh does this still work when its no longer sparse/freespace representation ?
g_sparse = [csr_matrix(dense) for dense in g_dense] # is it worth to convert to sparse matrix for dijkstra? 
goal_s = worlds.loc_to_state(worlds.target_x, worlds.target_y)
pred = np.array([dijkstra(g_sparse[i], indices=goal_s[i], return_predecessors=True)[1] for i in range(worlds.n_worlds)])
# pred = pred[:, 1]
cc = np.array([connected_components(g_sparse[i], directed=False, return_labels=True)[1] for i in range(worlds.n_worlds)])
cc_idx = [np.where(cc[i] == cc[i, goal_s[i]]) for i in range(worlds.n_worlds)]
start_x, start_y = worlds.get_coords(np.random.choice(cc_idx[1]))

plt.imshow(cc[0].reshape((worlds.n_row, worlds.n_col)), cmap="prism", alpha=0.6)
plt.imshow(pred[0].reshape((worlds.n_row, worlds.n_col)), cmap="prism", alpha=0.6)

for row in range(worlds.n_row):
    for col in range(worlds.n_col):
        state = worlds.loc_to_state(np.array([row]), np.array([col]))[0]
        number = pred[0, state] # Calculate the number for each cell
        text = ax.text(col, row, str(state), fontsize=9, color='red', ha='center', va='center')
ax.plot(worlds.target_x[0], worlds.target_y[0], 'bo')



plt.show()  # Make sure to display the plot if not in interactive mode
plt.draw()