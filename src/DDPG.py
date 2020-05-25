import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, smooth_l1_loss
import torch.optim as optim
from torch.distributions import Categorical

from collections import namedtuple
import random
import numpy as np

from ReplayMemory import ReplayMemory
from random_process import OrnsteinUhlenbeck

Transition = namedtuple('Transition', ['s', 'a', 'r', 's_'])

random.seed(0)

class actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(actor, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=state_dim, out_features=100),
            nn.Tanh())
        self.fc2 =  nn.Sequential(
            nn.Linear(in_features=100, out_features=action_dim),
            nn.Tanh())
        #self.sm = nn.Softmax(dim=-1)

    def forward(self, s):
        out1 = self.fc1(s)
        a = self.fc2(out1) # make each entry of `a` to be within [-1, 1]
        return a

class critic(nn.Module): # Q_w(s, a)
    def __init__(self, state_dim, action_dim):
        super(critic, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=state_dim+action_dim, out_features=100),
            nn.Tanh())
        self.fc2 = nn.Linear(in_features=100, out_features=1)

    def forward(self, s, a):
        out1 = self.fc1(torch.cat([s,a],1))
        Q = self.fc2(out1)
        return Q

class DDPG():
    def __init__(self, state_dim, action_dim, device, CONFIG):
        
        #== ENV PARAM ==
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        #== PARAM ==        
        self.LR_C = CONFIG.LR_C
        self.LR_C_START = CONFIG.LR_C
        self.LR_C_END = CONFIG.LR_C_END
        self.LR_C_DECAY = CONFIG.MAX_EP_STEPS * CONFIG.MAX_EPISODES / 2
        
        self.LR_A = CONFIG.LR_A
        self.LR_A_START = CONFIG.LR_A
        self.LR_A_END = CONFIG.LR_A_END
        self.LR_A_DECAY = CONFIG.MAX_EP_STEPS * CONFIG.MAX_EPISODES / 2
        
        self.BATCH_SIZE = CONFIG.BATCH_SIZE
        self.GAMMA = CONFIG.GAMMA
        self.MAX_MODEL = CONFIG.MAX_MODEL
        
        self.SIGMA = CONFIG.SIGMA

        #== CRITIC TARGET UPDATE PARAM ==
        self.double = CONFIG.DOUBLE
        self.TAU = CONFIG.TAU
        self.HARD_UPDATE = CONFIG.HARD_UPDATE
        self.SOFT_UPDATE = CONFIG.SOFT_UPDATE

        #== MODEL PARAM ==
        self.device = device
        
        #== MEMORY & MODEL ==
        self.memory = ReplayMemory(CONFIG.MEMORY_CAPACITY)
        self.build_network()

        self.random_process = OrnsteinUhlenbeck(action_dim, sigma=self.SIGMA, annealLen=CONFIG.MAX_EP_STEPS*2, dt=1)
        self.train = True

    def build_network(self):
        self.critic        = critic(self.state_dim, self.action_dim)
        self.critic_target = critic(self.state_dim, self.action_dim)
        self.actor         = actor(self.state_dim,  self.action_dim)
        self.actor_target  = actor(self.state_dim,  self.action_dim)
        if self.device == torch.device('cuda'):
            self.critic.cuda()
            self.critic_target.cuda()
            self.actor.cuda()
            self.actor_target.cuda()

        #== Optimizer ==
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.LR_C)
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=self.LR_A)
        self.training_step = 0

    def update_critic_target(self):
        if self.SOFT_UPDATE:
            #== Soft Replace ==
            for module_tar, module_pol in zip(self.critic_target.modules(), self.critic.modules()):
                if isinstance(module_tar, nn.Linear):
                    module_tar.weight.data = (1-self.TAU)*module_tar.weight.data + self.TAU*module_pol.weight.data
                    module_tar.bias.data   = (1-self.TAU)*module_tar.bias.data   + self.TAU*module_pol.bias.data
        elif self.training_step % self.HARD_UPDATE == 0:
            #== Hard Replace ==
            self.critic_target.load_state_dict(self.critic.state_dict())

    def update_actor_target(self):
        if self.SOFT_UPDATE:
            #== Soft Replace ==
            for module_tar, module_pol in zip(self.actor_target.modules(), self.actor.modules()):
                if isinstance(module_tar, nn.Linear):
                    module_tar.weight.data = (1-self.TAU)*module_tar.weight.data + self.TAU*module_pol.weight.data
                    module_tar.bias.data   = (1-self.TAU)*module_tar.bias.data   + self.TAU*module_pol.bias.data
        elif self.training_step % self.HARD_UPDATE == 0:
            #== Hard Replace ==
            self.actor_target.load_state_dict(self.actor.state_dict())

    def update(self):
        if len(self.memory) < self.BATCH_SIZE*20:
        #if not self.memory.isfull:
            return None, None
        self.training_step += 1
        
        #== EXPERIENCE REPLAY ==
        #transitions = self.memory.memory
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.s_)), 
                                      device=self.device, dtype=torch.bool)
        non_final_state_nxt = torch.FloatTensor([s for s in batch.s_ if s is not None], 
                                                 device=self.device)
        state = torch.FloatTensor(batch.s, device=self.device)
        action = torch.FloatTensor(batch.a, device=self.device).view(-1,1)
        reward = torch.FloatTensor(batch.r, device=self.device)
        
        #===================
        #== Update Critic ==
        #===================
        
        #== get Q_w (s,a) ==
        state_action_values = self.critic(state, action).view(-1)
        
        #== get a' = mu_theta' (s') ==
        #== get expected value: y = r + gamma * Q_w' (s',a') ==
        state_value_nxt = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            action_nxt = self.actor_target(non_final_state_nxt)
            Q_expect = self.critic_target(non_final_state_nxt, action_nxt).view(-1)
        state_value_nxt[non_final_mask] = Q_expect
        expected_state_action_values = (state_value_nxt * self.GAMMA) + reward

        self.critic.train()
        loss_critic = smooth_l1_loss(input=state_action_values, target=expected_state_action_values.detach())
        self.critic_opt.zero_grad()
        loss_critic.backward()
        self.critic_opt.step()
        
        #== Update Critic Target ==
        self.update_critic_target()
          
        #==================
        #== Update Actor ==
        #==================
        
        #== get a = mu_theta (s)
        action_policy = self.actor(state)
        
        self.actor.train()
        loss_actor = -self.critic(state, action_policy).mean()
        self.actor_opt.zero_grad()
        loss_actor.backward()
        self.actor_opt.step()

        #== Update Actor Target ==
        self.update_actor_target()

        #== Hyper-Parameter Update ==
        self.LR_C = self.LR_C_END + (self.LR_C_START - self.LR_C_END) * \
                                     np.exp(-1. * self.training_step / self.LR_C_DECAY)
        self.LR_A = self.LR_A_END + (self.LR_A_START - self.LR_A_END) * \
                                     np.exp(-1. * self.training_step / self.LR_A_DECAY)
        
        return loss_actor, loss_critic
    
    def select_action(self, state):
        state = torch.from_numpy(state).float()
        with torch.no_grad():
            tmp_a = self.actor(state).numpy()
            noise = self.random_process.sample()
            action = tmp_a + noise
        return action

    def store_transition(self, *args):
        self.memory.update(Transition(*args))