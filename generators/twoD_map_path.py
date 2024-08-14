from json.tool import main
from xml.sax.handler import DTDHandler
import scipy.sparse
from scipy.sparse import csr_matrix


import pickle
import numpy as np

import sys

sys.path.append('..')
sys.path.append('POMDP-Pairwise-Net/')
from generators.map_generator import MapGenerator
from domains.pomdp import POMDP



VIEWABLE_SIZE = 4

class Path():
    def __init__(self, states, actions, observations, bels):
        self.states = np.array(states, dtype = np.uint32)
        self.actions = np.array(actions, dtype = np.uint16)
        self.observations = np.array(observations, dtype = np.uint16)
        self.bels = [csr_matrix(dense_b) for dense_b in bels]   


class TwoDMapPath():
    def __init__(self, grid, goal_row, goal_column, solvers = ['QMDP']):
        self.grid = np.bool_(np.copy(grid))
        self.goal_r = int(goal_row)
        self.goal_c = int(goal_column)

        self.paths = {}
        for s in solvers:
            self.paths[s] = []
    
    def addPath(self, states, actions, observations, bels, solver_name):
        if solver_name not in self.paths.keys():
            print ("Solver not here")
        new_path = Path(states, actions, observations, bels) 
        self.paths[solver_name].append(new_path)
        
    def genPOMDP(self, discount = .99, T_noise = 0, O_noise = 0):
        ##### basics:
        room_rc = np.where(self.grid == 0) 
        states = ["s_" + str(x) + "_" + str(y) for x, y in zip(room_rc[0], room_rc[1])] 
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
                end_s_r = room_rc[0][k] + dr
                end_s_c = room_rc[1][k] + dc
                if self.grid[end_s_r, end_s_c] > 0:   ### obstacle
                    T[a, k, k] = 1
                else:   #### the end state is an adjacent room; so we need to find it in the list of empty rooms
                    end_ind = MapGenerator.rcToRoomIndex(self.grid, end_s_r, end_s_c)
                    T[a, k, end_ind] = 1

        ##### Reward
        R = np.zeros([num_A, num_S, num_S, num_O])  # (action, start_state, next_state, observation) 
        goal_ind = MapGenerator.rcToRoomIndex(self.grid, self.goal_r, self.goal_c)
        R[:, :, goal_ind, :] = 1

        #### Observation
        #### TODOL ADD OBS_noise
        Z = np.zeros([num_A, num_S, num_O])
        for k in range(num_S):
            r = room_rc[0][k]
            c = room_rc[1][k]
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
        return POMDP(states, actions, observations, T, Z, R , discount, prior) 

    @staticmethod
    def load_datasetBel(file_name, expert_solver = "Pairwise", path_len_max = 45, path_inds = np.array([0, 5], dtype = np.int64), max_num_path = 5000):
        twoD_map_file = open(file_name, "rb")
        twoD_maps = pickle.load(twoD_map_file)
        num_data = 300 * len(twoD_maps) * len(twoD_maps[0].paths[expert_solver])
        action_label = np.zeros(num_data, dtype = np.uint8)  
        success = np.ones(num_data, dtype = np.uint8)
        #states = np.zeros(num_data, dtype = np.uint8)  
        #observation = np.zeros([num_data, VIEWABLE_SIZE])  # We do not consider "GOAL" observation
        bel = np.zeros([num_data, twoD_maps[0].grid.shape[0], twoD_maps[0].grid.shape[1]])
        full_maps = np.zeros([num_data, twoD_maps[0].grid.shape[0], twoD_maps[0].grid.shape[1], 2], dtype = np.uint8)  #obstacles and goal (h in VINs) 

        main_index = 0
        for twoDmap in twoD_maps:
            room_rc = np.where(twoDmap.grid == 0) 
            for path in twoDmap.paths[expert_solver][:1]: 
                path_len = path.observations.size
                if path_len > path_len_max:
                    success[main_index] = 0
                instance = np.random.randint(path_len, size = 1)[0]
                full_maps[main_index, :, :, 0] = np.copy(twoDmap.grid)
                full_maps[main_index, twoDmap.goal_r, twoDmap.goal_c, 1] = 1
                action_label[main_index] = path.actions[instance]
                #states[main_index] = path.states[instance]  
                
                bel[main_index, room_rc[0], room_rc[1]] = path.bels[instance].toarray()  
                #obs_string = '{0:04b}'.format(path.observations[instance])  ### TODO:replace 4 with VIEWABLE_SIZ
                #for o_i in range(VIEWABLE_SIZE):
                #    observation[main_index, o_i] = int(obs_string[(VIEWABLE_SIZE-1) - o_i]) 
                main_index += 1
                if main_index == max_num_path:
                    return full_maps[:main_index], bel[:main_index], action_label[:main_index], success[:main_index] 
                    #observation_train[:main_index], states_train[:main_index] 
        print ("Num DATA:", main_index)
        return full_maps[:main_index], bel[:main_index], action_label[:main_index], success[:main_index] 



    @staticmethod
    def load_datasetBel_test(file_name, expert_solver = "Pairwise", path_len_max = 45):
        twoD_map_file = open(file_name, "rb")
        twoD_maps = pickle.load(twoD_map_file)
        num_data = len(twoD_maps) * 100 #len(twoD_maps[0].paths[expert_solver])
        states = np.zeros(num_data, dtype = np.uint8)  
        bel = np.zeros([num_data, twoD_maps[0].grid.shape[0], twoD_maps[0].grid.shape[1]])
        full_maps = np.zeros([num_data, twoD_maps[0].grid.shape[0], twoD_maps[0].grid.shape[1], 2], dtype = np.uint8)  #obstacles and goal (h in VINs) 
        success = np.ones(num_data, dtype = np.uint8)
        main_index = 0
        for twoDmap in twoD_maps:
            room_rc = np.where(twoDmap.grid == 0) 
            for path in twoDmap.paths[expert_solver][:3]: ### 50
                path_len = path.observations.size
                if path_len > path_len_max:
                    success[main_index] = 0

                full_maps[main_index, :, :, 0] = np.copy(twoDmap.grid)
                full_maps[main_index, twoDmap.goal_r, twoDmap.goal_c, 1] = 1
                states[main_index] = path.states[0]  
                bel[main_index, room_rc[0], room_rc[1]] = path.bels[0].toarray()  
                main_index += 1


        return states[:main_index], full_maps[:main_index], bel[:main_index], success[:main_index]


    @staticmethod
    def load_dataset(file_name, expert_solver = "Pairwise", train_ratio = .95, path_len_max = 45, num_steps = 4):
        twoD_map_file = open(file_name, "rb")
        twoD_maps = pickle.load(twoD_map_file)
        num_data = 300 * len(twoD_maps) * len(twoD_maps[0].paths[expert_solver])
        action_label = np.zeros([num_data, num_steps], dtype = np.uint8)  
        states = np.zeros(num_data, dtype = np.uint8)  
        observation = np.zeros([num_data, num_steps, VIEWABLE_SIZE])  # We do not consider "GOAL" observation
        bel = np.zeros([num_data, twoD_maps[0].grid.shape[0], twoD_maps[0].grid.shape[1]])
        full_maps = np.zeros([num_data, twoD_maps[0].grid.shape[0], twoD_maps[0].grid.shape[1], 2], dtype = np.uint8)  #obstacles and goal (h in VINs) 

        main_index = 0
        for twoDmap in twoD_maps:
            room_rc = np.where(twoDmap.grid == 0) 
            for path in twoDmap.paths[expert_solver]: ### 50
                path_len = path.observations.size - num_steps
                if path_len > path_len_max or path_len < 1:
                    #print ("Failed policy")
                    continue
                full_maps[main_index : main_index + path_len, :, :, 0] = np.copy(twoDmap.grid)
                full_maps[main_index : main_index + path_len, twoDmap.goal_r, twoDmap.goal_c, 1] = 1
                for st in range(num_steps):
                    action_label[main_index : main_index + path_len, st] = path.actions[st:path_len + st]
                states[main_index : main_index + path_len] = path.states[:path_len]  

                for step in range(path_len):
                    bel[main_index + step, room_rc[0], room_rc[1]] = path.bels[step].toarray()  
                    for st in range(num_steps):
                        obs_string = '{0:04b}'.format(path.observations[step + st])  ### TODO:replace 4 with VIEWABLE_SIZ
                        for o_i in range(VIEWABLE_SIZE):
                           observation[main_index + step, st, o_i] = int(obs_string[(VIEWABLE_SIZE-1) - o_i]) 
                main_index += path_len


        train_num = int(main_index * train_ratio)

        map_train = full_maps[:train_num]
        bel_train = bel[:train_num]
        action_train = action_label[:train_num]
        observation_train = observation[:train_num]
        states_train = states[:train_num]

        map_valid = full_maps[train_num: main_index]
        bel_valid = bel[train_num:main_index]
        action_valid = action_label[train_num:main_index]
        observation_valid = observation[train_num: main_index]
        states_valid = states[train_num:main_index]


        return states_train, map_train, bel_train, action_train, observation_train,\
        states_valid, map_valid, bel_valid, action_valid, observation_valid
