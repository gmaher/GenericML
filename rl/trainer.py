class Trainer:
    def __init__(self,algorithm,env,Nepisodes,Nsteps):
        """
        Initialize trainer object with chosen algorithm, environment and options
        """
        self.algorithm = algorithm
        self.env       = env
        self.N = Nepisodes
        self.Nsteps = Nsteps

    def train(self):

        algorithm.initialize()

        for i in range(self.N):
            iter_ = 0
            done = False
            s = self.env.reset()

            while not done and iter_ < self.Nsteps:

                a = algorithm.act(s)

                sprime,r,done,_ = self.env.step(a)

                algorithm.store((s,a,r,sprime,i,iter_))

                algorithm.update_step()

                iter_ += 1

            algorithm.update_episode()
