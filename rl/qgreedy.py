from algorithm import Algorithm
import numpy as np

class QGreedy(Algorithm):
    def __init__(self, model, dataBuffer, actionSpace):

        self.model         = model
        self.dataBuffer    = dataBuffer
        self.actionSpace = actionSpace

    def act(self,s):
        actions = self.actionSpace.actions()

        qList = []
        for a in actions:
            t = (s,a)
            q = self.model.predict(t)
            qList.append(q)

        qList = np.asarray(qList)
        inds = np.argmax(qList,axis=1)
        a = np.asarray([actions[i,inds[i]] for i in range(len(inds))])
        return a

    def store(self,tup):

        self.dataBuffer.append(tup)

    def update_step(self):

        tupList = self.dataBuffer.sample()
        self.model.train(tupList)
