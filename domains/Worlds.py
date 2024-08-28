# annoying things about the world I might wanna fix/change:
# only 4 actions + stay, not "in order"
# indexing is [a, s, s'] instead of [s, a, s']
# gridworld seems fancier with how it gets the transitions but this also makes more sense maybe...

import numpy as np 

class World():
    def __init__(self, grid, goal_r, goal_c):
        self.grid = grid
        self.goal_r = goal_r
        self.goal_c = goal_c
        self.n_rows = self.grid.shape[0]
        self.n_cols = self.grid.shape[1]
        self.room_rc = np.where(self.grid == 0)

        self.discount = 0.99  # um where this stored ... this is universal ?

        self.states, self.actions, self.observations, self.T, self.O, self.R, self.prior = self.genPOMDP()
        self.input_view = self.get_input_view()
        #  ^ inconsistent notation where genPOMDP uses Z but the constructor uses O

    # def genGoal(self, connection_percent_th = 80):
        # pass

    # is there benefit to making this a static method ? 
    def rcToRoomIndex(self, r, c):
        if self.grid[r, c]:
            print ("This cell is not empty!")
            return -1
        this_row = np.where(self.room_rc[0] == r)[0]
        this_row_column = np.where(self.room_rc[1][this_row] == c)[0]
        return this_row[0] + this_row_column[0]
    
    def roomIndexToRc(self, room_index):
        return self.room_rc[0][room_index], self.room_rc[1][room_index]
    
    @staticmethod
    def batchedRoomIndexToRc(grid, room_index):
        room_rc = [np.where(grid_i == 0) for grid_i in grid]
        room_r = []
        room_c = []
        for i in range(len(grid)):
            room_r.append(room_rc[i][0][int(room_index[i])])
            room_c.append(room_rc[i][1][int(room_index[i])])
        return room_r, room_c

    
    def getInputView(self):
        reward_mapping = -1 * np.ones(self.grid.shape) # -1 for freespace
        reward_mapping[self.goal_r, self.goal_c] = 10 # 10 at goal, if regenerating goals for each world then i'd need to redo this for each goal/trajectory.
        grid_view = np.reshape(self.grid, (1, 1, self.n_rows, self.n_cols))
        reward_view = np.reshape(reward_mapping, (1, 1, self.n_rows, self.n_cols))
        return np.concatenate((grid_view, reward_view), axis=1) # inlc empty 1 dim

    def genPOMDP(self, discount = .99, T_noise = 0, O_noise = 0):
        ##### basics:
        states = ["s_" + str(x) + "_" + str(y) for x, y in zip(self.room_rc[0], self.room_rc[1])] 
        actions = [(0, 1), (-1, 0), (0, -1), (1,0), (0, 0)] # ["right", "up", "left", "down", "stay"]
        observations = [f'{z:04b}' for z in range(0,16)] + ['G'] ### in the simplifed version of observation function we used 2^5 (considered center) for goal/no goal; make sure it works!        
        num_S  = len(states)
        num_A = len(actions)
        num_O = len(observations)

        ###### Transition function 
        ### TODO: ADD T_noise
        T = np.zeros([num_A, num_S, num_S]) ### action * start_state * end_state 
        for a in range(len(actions)):
            dr, dc = actions[a]
            for k in range(num_S): 
                end_s_r = self.room_rc[0][k] + dr
                end_s_c = self.room_rc[1][k] + dc
                if self.grid[end_s_r, end_s_c] > 0:   ### obstacle
                    T[a, k, k] = 1
                else:   #### the end state is an adjacent room; so we need to find it in the list of empty rooms
                    end_ind = self.rcToRoomIndex(end_s_r, end_s_c)
                    T[a, k, end_ind] = 1

        ##### Reward
        R = -1 * np.ones([num_A, num_S, num_S, num_O])  # make it so non-goal is punished
        goal_ind = self.rcToRoomIndex(self.goal_r, self.goal_c)
        R[:, :, goal_ind, :] = num_S # make reward proportional to num_S ??? not sure if thats useful haha

        #### Observation
        #### TODOL ADD OBS_noise
        Z = np.zeros([num_A, num_S, num_O])
        for k in range(num_S):
            r = self.room_rc[0][k]
            c = self.room_rc[1][k]
            o = 0
            for a in range(len(actions)):  #### using action coding as "head direction" for observations
                dr, dc = actions[a]
                o += (2 ** a) * int(self.grid[r + dr, c + dc])
            Z[:, k, o]  = 1
        ### terminal and goal states
        Z [:, goal_ind, :] = 0
        Z [:, goal_ind, -1] = 1
        
        prior = np.ones(num_S) / (num_S - 1)
        prior[goal_ind]  = 0
        return states, actions, observations, T, Z, R, prior

