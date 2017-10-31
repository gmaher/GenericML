import numpy as np

class Explorer:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace
    def explore(self,data,model):
        pass

class Greedy(Explorer):
    def explore(self,data):
        for a in self.actionSpace
