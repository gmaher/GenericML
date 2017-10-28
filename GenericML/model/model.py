from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib

class Model:
    def __init__(self):
        pass
    def predict(self,X):
        pass
    def train(self,X,Y):
        pass
    def save(self,fn):
        pass
    def load(self,fn):
        pass

class SKLearnModel(Model):
    def __init__(self,model_):
        self.model = model_
    def predict(self,X):
        self.predictions = self.model.predict(X)
        return self.predictions
    def train(self,X,Y):
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
        self.x = x_plh
        self.y = y_plh
        self.output_op = output_op
        self.train_op = train_op
        self.session = session
        self.saver = saver

    def predict(self,X):
        self.predictions = sess.run(self.output_op,
            {self.x:X})
        return self.predictions

    def train(self,X,Y):
        pass

    def trainStep(self,X,Y):
        self.sess.run(self.train_op,
            {self.x_plh:X,self.y_plh:Y})

    def save(self,fn):
        self.saver.save(self.session,fn)

    def load(self,fn):
        self.saver.restore(self.session,fn)
