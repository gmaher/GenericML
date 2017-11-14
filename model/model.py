from sklearn.linear_model import LinearRegression
import cPickle as pickle

def standardPreprocessor(tup):
    return tup[0]

def listPreprocessor(tupList):
    X = np.asarray([t[0] for t in tupList])
    Y = np.asarray([t[1] for t in tupList])
    return X,Y

class Model(object):
    def __init__(self,regressor):
        self.setPreprocessors()
        self.regressor = regressor
    def predict(self,data):
        X = self.predictPreprocessor(data)
        self.predictions = self.regressor.predict(X)
        return self.predictions
    def train(self,data):
        X,Y = self.trainPreprocessor(data)
        self.regressor.fit(X,Y)
    def setPreprocessors(self,predictPreprocessor=standardPreprocessor,
    trainPreprocessor=listPreprocessor):
        self.predictPreprocessor = predictPreprocessor
        self.trainPreprocessor   = trainPreprocessor
    def save(self,fn):
        pass
    def load(self,fn):
        pass

class SKLearnModel(Model):
    def save(self,fn):
        import pdb; pdb.set_trace()
        f = open(fn, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def load(self,fn):
        self.__dict__.update(pickle.loads(fn).__dict__)
        if self.model==None:
            raise RuntimeError('failed to load model at {}'.format(fn))

class TFModel(Model):
    def __init__(self,regressor,session,saver):
        super(TFModel,self).__init__(regressor)
        self.session = session
        self.saver = saver

    def save(self,fn):
        self.saver.save(self.session,fn)

    def load(self,fn):
        self.saver.restore(self.session,fn)

    def copy(self):
        self.regressor.copy()
