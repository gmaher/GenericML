class DataSet(object):
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
        self.data = []

    def append(self,d):
        self.data.append(d)

    def sample(self):
        return self.data

    def clear(self):
        self.data = []

class LookbackDataSet(ListDataSet):
    def __init__(self,lookback):
        super(LookbackDataSet,self).__init__()
        self.lookback = lookback

    def sample(self):
        return self.data[-self.lookback:]
