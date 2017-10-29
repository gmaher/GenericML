from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
import pdb
def standardPreprocessor(tup):
    return tup[0]

def listPreprocessor(tupList):
    X = np.asarray([t[0] for t in tupList])
    Y = np.asarray([t[1] for t in tupList])
    return X,Y

class Model(object):
    def __init__(self):
        self.setPreprocessors()
    def predict(self,data):
        pass
    def train(self,data):
        pass
    def setPreprocessors(self,predictPreprocessor=standardPreprocessor,
    trainPreprocessor=listPreprocessor):
        self.predictPreprocessor = predictPreprocessor
        self.trainPreprocessor   = trainPreprocessor
    def save(self,fn):
        pass
    def load(self,fn):
        pass

class SKLearnModel(Model):
    def __init__(self,model_):
        super(SKLearnModel,self).__init__()
        self.model = model_

    def predict(self,data):
        X = self.predictPreprocessor(data)
        self.predictions = self.model.predict(X)
        return self.predictions

    def train(self,data):
        X,Y = self.trainPreprocessor(data)
        self.model.fit(X,Y)

    def save(self,fn):
        joblib.dump(self.model,fn)

    def load(self,fn):
        self.model = joblib.load(fn)
        if self.model==None:
            raise RuntimeError('failed to load model at {}'.format(fn))

class TFModel(Model):
    def __init__(x_plh,y_plh,output_op,train_op,session,
        saver=None):
        super(TFModel,self).__init__()

        self.x = x_plh
        self.y = y_plh
        self.output_op = output_op
        self.train_op = train_op
        self.session = session
        self.saver = saver

    def predict(self,data):
        X = self.predictPreprocessor(data)
        self.predictions = sess.run(self.output_op,
            {self.x:X})
        return self.predictions

    def train(self,data):
        pass

    def trainStep(self,data):
        X,Y = self.trainPreprocessor(data)
        self.sess.run(self.train_op,
            {self.x_plh:X,self.y_plh:Y})

    def save(self,fn):
        self.saver.save(self.session,fn)

    def load(self,fn):
        self.saver.restore(self.session,fn)
