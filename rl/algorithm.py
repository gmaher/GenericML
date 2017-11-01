class Algorithm:
    def __init__(self):
        pass

    def initialize(self):
        pass

    def act(self,s):
        pass

    def store(self,tup):
        pass

    def update_step(self):
        pass

    def update_episode(self):
        pass

class SimpleRL(Algorithm):
    def __init__(self, model, explorer, dataBuffer):
        self.model         = model
        self.dataBuffer    = dataBuffer
        self.explorer      = explorer

    def act(self,data):
        a = self.explorer.explore(data,self.model)
        return a

    def store(self,tup):
        self.dataBuffer.append(tup)

    def update_step(self):
        trainingData = self.dataBuffer.sample()
        self.model.train(trainingData)
