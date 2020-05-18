# config.py
class config():
    def __init__(self,   MAX_EPISODES = 200, MAX_EP_STEPS = 200,
                         LR_A = 0.001, LR_C = 0.002, 
                         GAMMA = 0.9,     # reward discount
                         TAU = 0.01,      # soft replacement
                         MEMORY_CAPACITY = 10000,
                         BATCH_SIZE = 32,
                         RENDER = False,
                         ENV_NAME = 'Pendulum-v0',
                         EPSILON=1e-4,
                         MAX_MODEL=5):
                
        self.MAX_EPISODES = MAX_EPISODES
        self.MAX_EP_STEPS = MAX_EP_STEPS
        self.LR_A = LR_A
        self.LR_C = LR_C
        self.GAMMA = GAMMA
        self.TAU = TAU
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.BATCH_SIZE = BATCH_SIZE
        self.RENDER = RENDER
        self.ENV_NAME = ENV_NAME
        self.EPSILON = EPSILON
        self.MAX_MODEL= MAX_MODEL