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
