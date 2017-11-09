class Logger:
    def __init__(self,env,print_fn,render_fn,save_fn,n_print=100,n_render=10,n_save=10,theta=0.9):
        self.print_fn = print_fn
        self.render_fn = render_fn
        self.save_fn = save_fn
        self.n_print = n_print
        self.n_render = n_render
        self.n_saver = n_save
        self.env = env
        self.R = [0.0]
        self.theta = theta
    def log(self,data):
        episode = data[2]
        iter_ = data[3]
        done = data[4]
        r = data[1][2]
        if done:
            self.R.append(self.theta*self.R[-1]+(1-self.theta)*r)
        if episode % self.n_render == 0 and done:
            self.render_fn(self)
        if episode % self.n_saver == 0 and iter_==0:
            self.save_fn(self)
        if iter_%self.n_print == 0 or done:
            self.print_fn(self,data)

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

                tup = (data,newData,i,iter_,done)

                self.logger.log(tup)

                self.algorithm.store(tup)

                self.algorithm.update_step()

                data = newData

                iter_ += 1

            self.algorithm.update_episode()
