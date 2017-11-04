class Experiment(object):
    def __init__(self,initialize,train,finalize):
        self.initialize = initialize
        self.train = train
        self.finalize = finalize
