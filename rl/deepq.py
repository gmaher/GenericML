from algorithm import Algorithm
import numpy as np
class DeepQ(Algorithm):
    def __init__(self, model, target, explorer, replay_buffer, gamma, update_frequency):

        self.model         = model
        self.target_model  = target
        self.explorer      = explorer
        self.replay_buffer = replay_buffer
        self.gamma         = gamma
        self.update_frequency = update_frequency

        self.update_count = 0

    def initialize(self):
        self.update_count = 0

    def act(self,s):
        
        a = self.explorer.explore(s,self.model)
        return a

    def store(self,tup):

        self.replay_buffer.append(tup)

    def update_step(self):

        tup_list = self.replay_buffer.sample()
        data_list = [t[1] for t in tup_list]
        s    = np.asarray([t[1][0] for t in tup_list])
        r    = np.asarray([t[1][2] for t in tup_list])
        ss   = np.asarray([t[1][3] for t in tup_list])
        done = np.asarray([t[4] for t in tup_list])

        q_target = self.target_model.predict(ss)
        q_target = np.amax(q_target,axis=1)
        y = r + (~done)*self.gamma*q_target

        self.model.train((data_list,y))

        self.update_count += 1
        if self.update_count > self.update_frequency:
            self.target_model.copy()
            self.update_count = 0
