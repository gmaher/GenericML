import numpy as np

class Explorer:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace
    def explore(self,data,model):
        pass

class GreedyScalar(Explorer):
    def explore(self,data,model):
        for a in self.actionSpace.actions():
            q = getPrediction(data,a)
