class Regressor(object):
    def __init__(self):
        pass
    def predict(X):
        pass
    def fit(X,Y):
        pass

class TFRegressor(Regressor):
    def __init__(self,x_plh,y_plh,output_op,train_op,session,copy_op=None):
        self.x = x_plh
        self.y = y_plh
        self.output_op = output_op
        self.train = train_op
        self.session = session
        self.copy_op = copy_op
    def predict(self,X):
        return self.session.run(self.output_op,{self.x:X})
    def fit(self,X,Y):
        self.session.run(self.train, {self.x:X,self.y:Y})
    def copy(self):
        if self.copy_op == None:
            raise RuntimeError('No copy op specified')

        self.session.run(self.copy_op)
