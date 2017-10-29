class DataSet:
    def __init__(self):
        pass
    def append(self):
        pass
    def sample(self):
        pass
    def clear(self):
        pass

class ListDataSet(DataSet):
    def __init__(self):
        self.X = []
        self.Y = []

    def append(self,x,y):
        self.X.append(x)
        self.Y.append(y)

    def sample(self):
        return self.X,self.Y

    def clear(self):
        self.X = []
        self.Y = []

class LookbackDataSet(ListDataSet):
    def __init__(self,lookback):
        super(LookbackDataSet,self).__init__()
        self.lookback = lookback

    def sample(self):
        return self.X[-self.lookback:], self.Y[-self.lookback:]
