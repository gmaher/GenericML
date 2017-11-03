import pandas as pd
import numpy as np

class DataSet(object):
    def __init__(self):
        pass
    def append(self):
        pass
    def sample(self):
        pass
    def clear(self):
        pass

class NPArrayDataSet(DataSet):
    def __init__(self,X,Y, batch_mode=False, batch_size=32):
        self.X = X
        self.Y = Y
        self.Xshape = X.shape
        self.batch_mode = batch_mode
        self.batch_size = batch_size
    def sample():
        if not self.batch_mode:
            return (self.X,self.Y)
        else:
            ids = np.random.choice(self.Xshape[0],self.batch_size)
            return (X[ids],Y[ids])
    def clear():
        self.X = None
        self.Y = None

class DataframeDataSet(DataSet):
    def __init__(self,file_path):
        self.data = pd.read_csv(file_path)

    def sample():
        return self.data
        
    def clear():
        self.data = None

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
