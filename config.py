from modules.environments import LinearCovariate

from GenericML.model.model import SKLearnModel
from GenericML.rl.algorithm import SimpleRL
from GenericML.rl.trainer import Trainer
from GenericML.dataset.dataset import LookbackDataSet
from GenericML.rl.explore import GreedyVector
from GenericML.framework.experiment import Experiment

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

#Always put configure and finalize at the top
def configure(params):
    pspace = priceSpace(params["P_MIN"],
        params["P_MAX"],
        params["N_INTERVALS"],
        params["L_PERIOD"])

    regressor = LinearRegression()

    predictPreprocessor,trainPreprocessor = buildPreprocessors(params["END_COL"])

    model = LinearPricingModel(regressor)
    model.setPreprocessors(predictPreprocessor,trainPreprocessor)

    explorer = PriceExplorer(pspace)

    dataset = LookbackDataSet(lookback=params["LOOKBACK"])

    alg = SimpleRL(model,explorer,dataset)

    env = LinearCovariate(N=params["N_PERIODS"],
        n_covariates=params["N_COVARIATES"],
        alpha=params["ALPHA"],
        beta=params["BETA"],
        gamma=params["GAMMA"],
        sigma=params["SIGMA"],
        covRange=params["COV_RANGE"],
        l_period=params["L_PERIOD"])

    logger = Logger()

    def initialize():
        Tup,done = env.reset()
        s0 = Tup['s']
        p0 = np.random.rand(s0.shape[0])
        TupNew,done = env.step(p0)
        r = TupNew['r']
        X = np.concatenate((p0[:,np.newaxis],s0),axis=1)[:,:params['END_COL']]

        regressor.fit(X,r)

    trainer = Trainer(alg,env,logger,params["N_EPISODES"],params["N_PERIODS"])

    def finalize():
        R_obs = np.cumsum(logger.results['R_obs'])
        R_opt = np.cumsum(logger.results['R_opt'])

        plt.figure()
        plt.plot(R_obs/R_opt, linewidth=2,label='R_obs/R_obt')
        plt.legend()
        plt.show()

        t = np.arange(0,params["N_PERIODS"],1)+2

        regret = (R_opt-R_obs)/np.log(t)

        plt.figure()
        plt.plot(regret,linewidth=2,label='regret(t)/log(t)')
        plt.legend()
        plt.show()

    experiment = Experiment(initialize,trainer.train,finalize)

    return experiment

##############################################
# Defintitions
##############################################

class priceSpace:
    def __init__(self,pmin,pmax,intervals,N):
        self.pmin = pmin
        self.pmax = pmax
        self.intervals = intervals
        self.N = N

    def actions(self):
        p = np.arange(self.pmin,self.pmax,(1.0*self.pmax-self.pmin)/self.intervals)
        Prices = np.zeros((self.N,self.intervals))
        Prices[:] = p
        return Prices

class PriceExplorer(GreedyVector):
    def getPrediction(self,data,model):
        A = self.actionSpace.actions()
        s = data['s']
        Q = np.zeros_like(A)
        for i in range(A.shape[1]):
            a = A[:,i]
            q = model.predict({"s":s,"a":a})
            Q[:,i] = q
        return Q,A

def buildPreprocessors(END_COL):

    def predictPreprocessor(data):
        s = data['s']
        p = data['a']

        X = np.concatenate((p[:,np.newaxis],s),axis=1)
        return X[:,:END_COL]

    def trainPreprocessor(data):
        """data = [(D,D',),...]"""
        Xlist = [predictPreprocessor({"s":t[0]['s'],"a":t[1]['a']}) for t in data]
        X = np.concatenate(Xlist)
        if X.shape[0] == 1: X = X[0,:END_COL]

        Ylist = [t[1]['r'] for t in data]
        Y = np.concatenate(Ylist)
        if Y.shape[0] == 1: Y = Y[0,:END_COL]

        return X[:,:END_COL],Y
    return predictPreprocessor, trainPreprocessor

class LinearPricingModel(SKLearnModel):
    def predict(self,data):
        D = super(LinearPricingModel,self).predict(data)
        P = data['a']
        R = D*P

        return R.copy()

class Logger:
    def __init__(self):
        self.results = {}
        self.results['R_obs'] = []
        self.results['R_opt'] = []
    def log(self,data):
        iter_ = data[3]
        robs = data[1]['R_obs']
        ropt = data[1]['R_opt']
        self.results['R_obs'].append(robs)
        self.results['R_opt'].append(ropt)

        if iter_%100 == 0:
            print "Iteration {}: R_obs = {}, R_opt = {}, R_obs/R_opt = {}".format(
                iter_,robs,ropt,1.0*robs/ropt
            )
