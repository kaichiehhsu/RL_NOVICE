import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, smooth_l1_loss
import torch.optim as optim
from torch.distributions import Categorical

from collections import namedtuple
import random
import numpy as np

from ReplayMemory import ReplayMemory

Transition = namedtuple('Transition', ['s', 'a', 'r', 's_'])

ActorMem = namedtuple('ActorMem', ['log_p', 'value'])

random.seed(0)

class actor(nn.Module):
    def __init__(self, state_num, action_num):
        super(actor, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=state_num, out_features=128),
            nn.Tanh())
        self.fc4 = nn.Linear(in_features=128, out_features=action_num)
        self.sm = nn.Softmax(dim=-1)

    def forward(self, x):
        out1 = self.fc1(x)
        a = self.fc4(out1)
        probs = self.sm(a)
        return probs

class critic(nn.Module):
    def __init__(self, state_num, action_num):
        super(critic, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=state_num, out_features=128),
            nn.Tanh())
        self.fc4 = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        out1 = self.fc1(x)
        v = self.fc4(out1)
        return v

class Actor_Critic():
    def __init__(self, state_num, action_num, device, CONFIG, action_list):
        
        self.action_list = action_list
        self.memory = ReplayMemory(CONFIG.MEMORY_CAPACITY)
        self.actor_memory = ReplayMemory(CONFIG.MEMORY_CAPACITY)
        
        #== ENV PARAM ==
        self.state_num = state_num
        self.action_num = action_num
        
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

        #== CRITIC TARGET UPDATE PARAM ==
        self.TAU = CONFIG.TAU
        self.HARD_UPDATE = CONFIG.HARD_UPDATE
        self.SOFT_UPDATE = CONFIG.SOFT_UPDATE
        
        #== DQN ==
        self.double = CONFIG.DOUBLE
        self.device = device
        self.build_network()
        self.max_grad_norm = 1.0

    def build_network(self):
        self.critic        = critic(self.state_num, self.action_num)
        self.critic_target = critic(self.state_num, self.action_num)
        self.actor         = actor(self.state_num,  self.action_num)
        if self.device == torch.device('cuda'):
            self.critic.cuda()
            self.critic_target.cuda()

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
    
    def update_critic(self):
        if len(self.memory) < self.BATCH_SIZE*10:
        #if not self.memory.isfull:
            return
        
        #== EXPERIENCE REPLAY ==
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.s_)), 
                                      device=self.device, dtype=torch.bool)
        non_final_state_nxt = torch.FloatTensor([s for s in batch.s_ if s is not None], 
                                                 device=self.device)
        state = torch.FloatTensor(batch.s, device=self.device)
        action = torch.LongTensor(batch.a, device=self.device).view(-1,1)
        reward = torch.FloatTensor(batch.r, device=self.device)
        
        #== get V(s) ==
        state_values = self.critic(state).view(-1)
        
        #== get expected value: y = r + gamma * V(s') ==
        state_value_nxt = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            if self.double:
                Q_expect = self.critic_target(non_final_state_nxt)
            else:
                Q_expect = self.critic(non_final_state_nxt)
        state_value_nxt[non_final_mask] = Q_expect.view(-1)
        expected_state_values = (state_value_nxt * self.GAMMA) + reward
        
        #== regression V(s) -> y ==
        self.critic.train()
        loss_critic = smooth_l1_loss(input=state_values, target=expected_state_values.detach())
        
        #== backward optimize ==
        self.critic_opt.zero_grad()
        loss_critic.backward()
        self.critic_opt.step()
        
        #== Update Target Network ==
        self.update_critic_target()
        
        #== Hyper-Parameter Update ==
        self.LR_C = self.LR_C_END + (self.LR_C_START - self.LR_C_END) * \
                                     np.exp(-1. * self.training_step / self.LR_C_DECAY)

        return loss_critic.item()
    
    def update_actor(self, reward_record):

        R=0
        returns=[]
        for r in reward_record[::-1]:
            # calculate the discounted value
            R = r + self.GAMMA * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss_actor=[]
        for (log_prob, value), R in zip(self.actor_memory.memory, returns):
            advantage = R - value.item()

            loss_actor.append(-log_prob * advantage)

        loss_actor = torch.stack(loss_actor).mean()
        
        #== Update Actor ==
        self.actor_opt.zero_grad()
        loss_actor.backward()
        self.actor_opt.step()

        #== Hyper-Parameter Update ==
        self.LR_A = self.LR_A_END + (self.LR_A_START - self.LR_A_END) * \
                                     np.exp(-1. * self.training_step / self.LR_A_DECAY)

        return loss_actor.item()
    
    def update(self,reward_record):
        self.training_step += 1
        
        loss_critic = self.update_critic()
        loss_actor = self.update_actor(reward_record)

        return loss_actor, loss_critic
    '''    
    def update(self, reward_record):
        self.training_step += 1

        R=0
        returns=[]
        for r in reward_record[::-1]:
            # calculate the discounted value
            R = r + self.GAMMA * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss_actor=[]
        loss_critic=[]
        for (log_prob, value), R in zip(self.actor_memory.memory, returns):
            advantage = R - value.item()

            loss_actor.append(-log_prob * advantage)

            loss_critic.append(smooth_l1_loss(value, torch.tensor([R])))

        loss_actor = torch.stack(loss_actor).mean()
        loss_critic = torch.stack(loss_critic).mean()

        #== Update Critic ==
        self.critic_opt.zero_grad()
        loss_critic.backward()
        self.critic_opt.step()
        
        #== Update Actor ==
        self.actor_opt.zero_grad()
        loss_actor.backward()
        self.actor_opt.step()

        #== Hyper-Parameter Update ==

        self.LR_A = self.LR_A_end + (self.LR_A_start - self.LR_A_end) * \
                                     np.exp(-1. * self.training_step / self.LR_A_decay)
        self.LR_C = self.LR_C_end + (self.LR_C_start - self.LR_C_end) * \
                                     np.exp(-1. * self.training_step / self.LR_C_decay)

        return loss_actor.item(), loss_critic.item()
    '''
    
    def select_action(self, state):
        self.critic.train()
        self.actor.train()

        state = torch.from_numpy(state).float()
        probs = self.actor(state)
        pi = Categorical(probs)

        action_index = pi.sample()

        value = self.critic(state)
        log_p = pi.log_prob(action_index)

        self.actor_memory.update(ActorMem(log_p, value))
        
        return self.action_list[action_index], action_index.item()

    def store_transition(self, *args):
        self.memory.update(Transition(*args))