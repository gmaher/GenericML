from algorithm import Algorithm

class SupervisedBandit(Algorithm):
    def __init__(self, model, dataBuffer):

        self.model         = model
        self.dataBuffer    = dataBuffer

    def act(self,s):

        a = self.model.predict(s)
        return a

    def store(self,tup):

        self.dataBuffer.append(tup)

    def update_step(self):

        tup = self.dataBuffer.sample()

        X    = tup[0]
        Y    = tup[2]

        self.model.train(s,y)
