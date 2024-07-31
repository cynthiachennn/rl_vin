from dataset.dataset import *
from dataset.make_training_data import *
from domains.gridworld import *
from generators.obstacle_gen import *
from model import *

# np.random.seed(9)

### functions that make up the underlying code
# enerate data
def generate_gridworld(max_obs, dom_size):
    border_res = False
    while not border_res:
        goal = [np.random.randint(dom_size[0]), np.random.randint(dom_size[1])]
        obs = obstacles(dom_size, goal, max_obs) # can use obs.show() to show obs map - black/0 = free, white/1 = obstacle...
        n_obs = obs.add_n_rand_obs(max_obs)
        border_res = obs.add_border()
    im = obs.get_final()
    G = GridWorld(im, goal[0], goal[1])
    return G

# get an example trajectory (dont use this anymore?)
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

# visualize gridworld and calculated paths if any
def visualize(G, start=None, goal=None, targ_traj=None, pred_traj=None, opt_traj=None): # ugh sorta redundant but not worth cleaning up
    # plt.ion()
    fig, ax = plt.subplots()
    implot = plt.imshow(G.image.T, cmap='Greys_r') # WHY T
    if start is not None:
        ax.plot(start[0], start[1], 'ro', label='Start')
    if goal is not None:
        ax.plot(goal[0], goal[1], 'go', label='Goal')
    if targ_traj is not None:
        ax.plot(targ_traj[:, 0], targ_traj[:, 1], c='b', label='Optimal Path')
    if pred_traj is not None:
        ax.plot(pred_traj[:, 0], pred_traj[:, 1], c='r', label='Predicted Path')
    if opt_traj is not None:
        ax.plot(opt_traj[:, 0], opt_traj[:, 1], c='g', label='Djikstra Path')
    plt.show()

# get optimal trajectory from start to goal
def get_trajectory(G, start, goal): # start and goal are state val not coords
    _, W = G.get_graph_inv()
    path = []
    g_dense = W
    g_masked = np.ma.masked_values(g_dense, 0)
    g_sparse = csr_matrix(g_dense)
    d, pred = dijkstra(g_sparse, indices=goal, return_predecessors=True)
    states = trace_path(pred, goal, start) # what is pred oh nvm its the djkstra vals ok
    states = np.flip(states, 0)# .reshape(states.shape[0])
    for state in states:
        # print(state.shape)
        r, c = G.get_coords(np.int64(state[0]))
        path.append((r, c))
    return path

def train_loop(config, G, agent):
    episodes = 200
    max_steps = 50 # steps we wanna try before we give up on finding goal (computational bound)
    total_steps = 0

    q_target = torch.zeros((config.imsize, config.imsize, 8))
    agent.gamma = 0.75

    for ep in range(episodes):
        current_state = np.int64(np.random.randint(G.G.shape[0]))
        done = False
        agent.learn_world(G)
        for step in range(max_steps):
            total_steps = total_steps + 1
            action = agent.compute_action(G.get_coords(current_state))
            next_state = G.sample_next_state(current_state, action)
            reward = G.R[current_state][action]
            state_x, state_y = G.get_coords(current_state)
            state_x_, state_y_ = G.get_coords(next_state)
            q_target[state_x][state_y][action] = reward + agent.gamma * max(q_target[state_x_][state_y_])
            if next_state == G.map_ind_to_state(G.target_x, G.target_y): 
                done = True
            agent.store_episode((state_x, state_y), action, reward, (state_x_, state_y_), done)
            if done == True:
                agent.update_exploration_prob(decay=0.001)
                break
            current_state = next_state
        if total_steps >= config.batch_size: # should we still train if goal was never reached ? is that useful.
            print('training...')
            agent.train(config.batch_size, 
                        criterion=nn.MSELoss(),  # used to be Cross Entropy, but I think that works better for multiclass/not good for predicing specific values. 
                        optimizer=optim.RMSprop(agent.model.parameters(), 
                                lr=config.lr, eps=1e-6), q_target=q_target)
            total_steps = 0 # reset total steps so it can rebuild memory buffer??? im not sure.
            agent.memory_buffer = [] # and reset memory buffer ? not sure. 
    return agent, q_target

# calculate policy (argmax) of a matrix of q values generated from NN
def get_policy(agent, q_target):
    q_values = agent.q_values
    pred_q = [agent.model.fc(q_values[0, :, i, j]).detach().numpy() for i in range(agent.config.imsize) for j in range(agent.config.imsize)]
    # q_final = [agent.model.fc(q_values[0, :, j, i]).detach().numpy() for i in range(config.imsize) for j in range(config.imsize)]
    pred_actions = np.argmax(pred_q, axis=1)
    # format like the grid
    pred_actions = np.array([pred_actions[i:i+agent.config.imsize] for i in range(0, len(pred_actions), agent.config.imsize)])
    target_actions = np.argmax(q_target, axis=2)
    print(pred_actions)
    print(target_actions.detach().numpy())
    return pred_actions, target_actions
        
# generate a random valid start coordinate and the path from it to the goal
def generate_path(G):
    dijkstra_traj = None
    count = 0
    while not dijkstra_traj and count < 50:
        start_state = np.int64(np.random.randint(G.G.shape[0]))
        dijkstra_traj = get_trajectory(G, start_state, G.map_ind_to_state(G.target_x, G.target_y))
        if start_state == G.map_ind_to_state(G.target_x, G.target_y):
            dijkstra_traj = False
        count += 1
    return dijkstra_traj, start_state

# use predicted q values from neural network to generate a path 
def get_pred_path(start_state, G, agent):
    pred_traj = []
    current_state = start_state
    done = False
    steps = 0
    agent.exploration_prob = 0 # follow policy explicitly now.
    while not done and steps < agent.config.imsize**2:
        pred_traj.append(G.get_coords(current_state))
        # print('current state', G.get_coords(current_state))
        action = agent.compute_action(G.get_coords(current_state)) # this recalculates, should i just get from agent.q_values? should i store agent.actions somewhere?
        # especially since i never use q_values without the fc layer. and barely use them without argmax (just for loss)
        # action = np.argmax(q_values[current_state])
        # print('action', action)
        next_state = G.sample_next_state(current_state, action)
        if next_state == G.map_ind_to_state(G.target_x, G.target_y):
            done = True
            pred_traj.append(G.get_coords(next_state))
            # print('solved!')
        current_state = next_state
        steps += 1
    # print('pred', steps)
    # if done == False:
        # print('failed :(')
    return pred_traj, done

# use the target values calculated from agent's experience to generate a path 
# the target values are not always correct so it's different from the optimal path sometimes
def get_target_path(start_state, G, agent, target_actions):
    targ_traj = []
    done = False
    current_state = start_state
    steps = 0
    while not done and steps < agent.config.imsize**2:
        targ_traj.append(G.get_coords(current_state))
        action = target_actions[G.get_coords(current_state)[0], G.get_coords(current_state)[1]]
        next_state = G.sample_next_state(current_state, action)
        if next_state == G.map_ind_to_state(G.target_x, G.target_y):
            done = True
            targ_traj.append(G.get_coords(next_state))
        current_state = next_state
        steps +=1

   # if done == False:
        # print('target failed :(')
    # print('target steps:', steps)
    return targ_traj, done


# contains all the agent's internal knowledge ?
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
            # print('random')
            return np.random.choice(range(self.n_actions))
        # q_values = self.model.forward(self.state)
        # print('policy')
        # q_values = self.model.map_qs(self.iv_mixed, self.config.k)
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
            
    # with q_target precalcuated !!!!
    def train(self, batch_size, criterion, optimizer, q_target):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # idk just adding this cuz
        # sample from memory buffer
        batch = np.random.permutation(self.memory_buffer)
        batch = batch[:batch_size]
        avg_loss = []
        # q_target = np.zeros((self.config.imsize, self.config.imsize, 8))

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
            # q_target = experience['reward'] + self.gamma * np.max(q_target[experience['next_state']])
            # also not sure if i should redo q_target during training but i dont think so..
            pred, output = self.model.fc(q_values[0, :, state_x, state_y]), q_target[state_x][state_y]
            # print(pred[experience['action']], output[experience['action']])
            
            loss = criterion(pred, output)
            avg_loss.append(loss.item())
            loss.backward()
            optimizer.step() 
        print('average loss:', np.mean(avg_loss))
            
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

def get_q_target(agent, G):
    episodes = 200
    max_steps = 500
    q_target = np.zeros((G.G.shape[0], 8)) # q val array = states x actions ?
    agent.gamma = 0.75

    def step(current_state, q_target):
        action = agent.compute_action(G.get_coords(current_state))
        next_state = G.sample_next_state(current_state, action)
        reward = G.R[current_state][action]
        q_target[current_state][action] = reward + agent.gamma * max(q_target[next_state])
        # print('current state:', current_state)
        # print('action:', action)
        # print('reward', reward)
        # print('q_value', q_values[current_state][action])
        return next_state, q_target

    for i in range(episodes):
        current_state = np.int64(np.random.randint(G.G.shape[0]))
        for j in range(max_steps):
            next_state, q_target = step(current_state, q_target)
            current_state = next_state
            if current_state == G.map_ind_to_state(G.target_x, G.target_y):
                # print('goal reached!')
                break

    for state in range(G.G.shape[0]):
        print('state:', state)
        print(q_target[state])

    [np.argmax(state) for state in q_target]

    return q_target
            
         
