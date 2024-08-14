"""

Loading POMDP environment files.

author: mbforbes
some additions: koosha

"""

import numpy as np


class POMDP():
    def __init__(self, states, actions, observations, T, O, R , discount, prior):
        self.states = states
        self.actions = actions
        self.observations = observations
        self.n_states = len(states)
        self.n_actions = len(actions)
        self.T = np.copy(T)
        self.O = np.copy(O)
        self.R = np.copy(R)
        self.discount = discount
        self.prior = np.copy(prior) 
    

    def print_summary(self):
        print ("discount:", self.discount)
        print ("states:", self.states)
        print ("actions:", self.actions)
        print ("observations:", self.observations)
        print ("")
        print ("T:", self.T)
        print ("")
        print ("O:", self.O)
        print ("")
        print ("R:", self.R)



