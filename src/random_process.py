import numpy as np

class OrnsteinUhlenbeck():
    def __init__(self, dim=1, theta=0.1, mu=0., sigma=0.2, dt=1e-2, x0=0., sigma_min=1e-2, annealLen = 1000):
                
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.dim = dim
        self.x0 = x0
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.annealLen = annealLen
        
        self.reset()
        
    def cur_sigma(self):
        maxLen = np.minimum(self.annealLen, self.step)
        return self.sigma - maxLen * self.sigma_space

    def sample(self):
        #print(self.cur_sigma(),end=', ')
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.cur_sigma() * np.sqrt(self.dt) * np.random.randn(self.dim)

        self.x_prev = x
        self.step += 1

        return x

    def reset(self):
        self.step = 0
        self.x_prev = np.ones(shape=self.dim) * self.x0
        self.sigma_space = (self.sigma - self.sigma_min) / self.annealLen