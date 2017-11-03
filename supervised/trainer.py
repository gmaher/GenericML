class Trainer(object):
    def __init__(self,model,dataset):
        self.model   = model
        self.dataset = dataset
    def train(self):
        trainingData = self.dataset.sample()
        self.model.train(trainingData)

class BatchTrainer(Trainer):
    def train(self,iterations):
        for i in range(iterations):
            trainingData = self.dataset.sample()
            self.model.train(trainingData)
