class Trainer:
    def __init__(self,algorithm,env,logger,Nepisodes,Nsteps):
        """
        Initialize trainer object with chosen algorithm, environment and options
        """
        self.algorithm = algorithm
        self.env       = env
        self.logger    = logger
        self.N = Nepisodes
        self.Nsteps = Nsteps

    def train(self):

        self.algorithm.initialize()

        for i in range(self.N):
            iter_ = 0
            done = False
            data,done = self.env.reset()

            while not done and iter_ < self.Nsteps:

                a = self.algorithm.act(data)

                newData,done = self.env.step(a)

                self.logger.log((data,newData,i,iter_))

                self.algorithm.store((data,newData,i,iter_))

                self.algorithm.update_step()

                data = newData

                iter_ += 1

            self.algorithm.update_episode()
