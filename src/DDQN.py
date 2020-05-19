import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, smooth_l1_loss
from torch.autograd import Variable
import torch.optim as optim

import random
import numpy as np

from model import model
from ReplayMemory import ReplayMemory, Transition

class DDQN():
    def __init__(self, state_num, action_num, device, CONFIG):
        
        self.action_list = np.linspace(-2, 2, num=action_num, endpoint=True).reshape(-1,1)
        print(self.action_list.reshape(-1))
        self.memory = ReplayMemory(CONFIG.MEMORY_CAPACITY)
        
        #== ENV PARAM ==
        self.state_num = state_num
        self.action_num = action_num
        
        #== PARAM ==
        self.epsilon = CONFIG.EPSILON
        self.eps_start = CONFIG.EPSILON
        self.eps_end = CONFIG.EPSILON_END
        self.decay = CONFIG.MAX_EP_STEPS
        
        self.LR_A = CONFIG.LR_A
        self.LR_A_start = CONFIG.LR_A
        self.LR_A_end = CONFIG.LR_A_END
        self.LR_A_decay = CONFIG.MAX_EP_STEPS * CONFIG.MAX_EPISODES / 2
        
        self.BATCH_SIZE = CONFIG.BATCH_SIZE
        self.GAMMA = CONFIG.GAMMA
        self.MAX_MODEL = CONFIG.MAX_MODEL
        self.TAU = CONFIG.TAU
        
        #== DQN ==
        self.double = CONFIG.DOUBLE
        self.device = device
        self.build_network()

    def build_network(self):
        self.Q_network = model(self.state_num, self.action_num)
        self.target_network = model(self.state_num, self.action_num)
        if self.device == torch.device('cuda'):
            self.Q_network.cuda()
            self.target_network.cuda()
        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=self.LR_A)
        self.max_grad_norm = 0.5
        self.training_step = 0
    
    def update_target_network(self):
        # Q_target <- Q_policy
        #self.target_network.load_state_dict(self.Q_network.state_dict())
        #== soft update ==
        for module_tar, module_pol in zip(self.target_network.modules(), self.Q_network.modules()):
            if isinstance(module_tar, nn.Linear):
                module_tar.weight.data = (1-self.TAU)*module_tar.weight.data + self.TAU*module_pol.weight.data
                module_tar.bias.data   = (1-self.TAU)*module_tar.bias.data   + self.TAU*module_pol.bias.data
         
    def update_Q_network(self):
        if len(self.memory) < self.BATCH_SIZE*20:
        #if not self.memory.isfull:
            return
        self.training_step += 1
        
        #== EXPERIENCE REPLAY ==
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.s_)), 
                                      device=self.device, dtype=torch.bool)
        non_final_state_nxt = torch.FloatTensor([s for s in batch.s_
                                                     if s is not None])
        state = torch.FloatTensor(batch.s)
        action = torch.LongTensor(batch.a).view(-1,1)
        reward = torch.FloatTensor(batch.r)
        
        if self.device == torch.device('cuda'):
            state = state.cuda()
            action = action.cuda()
            non_final_state_nxt = non_final_state_nxt.cuda()
            reward = reward.cuda()
        
        #== get Q(s,a) ==
        # gather reguires idx to be Long, i/p and idx should have the same shape with only diff at the dim we want to extract value
        # o/p = Q [ i ][ action[i] ], which has the same dim as idx, 
        state_action_values = self.Q_network(state).gather(1, action).view(-1)
        
        #== get a' by Q_policy: a' = argmax_a' Q_policy(s', a') ==
        with torch.no_grad():
            action_nxt = self.Q_network(non_final_state_nxt).max(1, keepdim=True)[1]
        
        #== get expected value: y = r + gamma * Q_tar(s', a') ==
        state_value_nxt = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            if self.double:
                Q_tmp = self.target_network(non_final_state_nxt)
            else:
                Q_tmp = self.Q_network(non_final_state_nxt)
        state_value_nxt[non_final_mask] = Q_tmp.gather(1, action_nxt).view(-1)
        expected_state_action_values = (state_value_nxt * self.GAMMA) + reward
        
        #== regression Q(s, a) -> y ==
        self.Q_network.train()
        loss = smooth_l1_loss(input=state_action_values, target=expected_state_action_values.detach())
        
        #== backward optimize ==
        self.optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_norm_(self.Q_network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        #== Hard Replace ==
        if self.training_step % 200 == 0:
            self.target_network.load_state_dict(self.Q_network.state_dict())
        #self.update_target_network()
        
        #== Hyper-Parameter Update ==
        self.epsilon = self.eps_end + (self.eps_start - self.eps_end) * \
                                       np.exp(-1. * self.training_step / self.decay)
        self.LR_A = self.LR_A_end + (self.LR_A_start - self.LR_A_end) * \
                                     np.exp(-1. * self.training_step / self.LR_A_decay)

        return state_action_values.mean().item()

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        if np.random.random() < self.epsilon:
            action_index = np.random.randint(self.action_num)
        else:
            action_index = self.Q_network(state).max(1)[1].item()
        return self.action_list[action_index], action_index
    
    def store_transition(self, *args):
        self.memory.update(Transition(*args))

    def save(self, step, logs_path):
        os.makedirs(logs_path, exist_ok=True)
        model_list =  glob.glob(os.path.join(logs_path, '*.pth'))
        if len(model_list) > self.MAX_MODEL - 1 :
            min_step = min([int(li.split('/')[-1][6:-4]) for li in model_list]) 
            os.remove(os.path.join(logs_path, 'model-{}.pth' .format(min_step)))
        logs_path = os.path.join(logs_path, 'model-{}.pth' .format(step))
        self.Q_network.save(logs_path, step=step)
        print('=> Save {}' .format(logs_path)) 
    
    def restore(self, logs_path):
        self.Q_network.load(logs_path)
        self.target_network.load(logs_path)
        print('=> Restore {}' .format(logs_path))