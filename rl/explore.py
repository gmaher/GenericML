import numpy as np

class Explorer:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace
    def explore(self,data,model):
        pass
    def getPrediction(self,data,model):
        raise RuntimeError('getPrediction not implemented')

class GreedyVector(Explorer):
    def explore(self,data,model):
        """Q,A = (Nbatch x Nactions)"""
        Q,A = self.getPrediction(data,model)
        inds = np.argmax(Q,axis=1)
        a = np.asarray([A[i,inds[i]] for i in range(len(inds))])
        return a

class EpsilonGreedy(Explorer):
    def __init__(self,epsilon=1.0, decay=0.99, n=100):
        self.epsilon = epsilon
        self.decay = decay
        self.n = n
        self.count=0
    def explore(self,data,model):
        self.count+=1
        if self.count > self.n:
            self.count = 0
            self.epsilon *= self.decay
        
        Q = model.predict(data)
        n_actions = Q.shape[1]
        a = np.argmax(Q,axis=1)[0]
        r = np.random.rand()
        if r < self.epsilon:
            a = np.random.randint(n_actions)

        return a
