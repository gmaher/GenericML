from algorithm import Algorithm

class SupervisedBandit(Algorithm):
    def __init__(self, model, dataBuffer):

        self.model         = model
        self.dataBuffer    = dataBuffer

    def act(self,tup):

        a = self.model.predict(tup)
        return a

    def store(self,tup):

        self.dataBuffer.append(tup)

    def update_step(self):

        tupList = self.dataBuffer.sample()

        self.model.train(tupList)
