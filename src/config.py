# config.py
class config():
    def __init__(self,  ENV_NAME='Pendulum-v0',
                        MAX_EPISODES=200, MAX_EP_STEPS=200,
                        LR_A=1e-3, LR_A_END=1e-4,
                        LR_C=1e-3, LR_C_END=1e-4,
                        EPSILON=0.9, EPSILON_END=5e-2,
                        GAMMA=0.9, 
                        TAU=0.01, HARD_UPDATE=200, SOFT_UPDATE=False,
                        MEMORY_CAPACITY=10000,
                        BATCH_SIZE=32,
                        RENDER=False,                         
                        MAX_MODEL=5,
                        DOUBLE=True):
                
        self.MAX_EPISODES = MAX_EPISODES
        self.MAX_EP_STEPS = MAX_EP_STEPS
        
        self.LR_A = LR_A
        self.LR_A_END = LR_A_END
        self.LR_C = LR_C
        self.LR_C_END = LR_C_END
        self.EPSILON = EPSILON
        self.EPSILON_END = EPSILON_END
        
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        
        self.TAU = TAU
        self.HARD_UPDATE = HARD_UPDATE
        self.SOFT_UPDATE = SOFT_UPDATE
        
        self.RENDER = RENDER
        self.ENV_NAME = ENV_NAME
        
        self.MAX_MODEL= MAX_MODEL
        
        self.DOUBLE = DOUBLE