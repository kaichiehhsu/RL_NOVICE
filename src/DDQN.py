import torch
from torch.nn.functional import mse_loss
from torch.autograd import Variable
import torch.optim as optim

import random
import numpy as np

from model import model

class DDQN():
    def __init__(self, state_num, action_num, device, CONFIG):
        #== ENV PARAM ==
        self.state_num = state_num
        self.action_num = action_num
        
        #== PARAM ==
        self.epsilon = CONFIG.EPSILON
        self.BATCH_SIZE = CONFIG.BATCH_SIZE
        self.GAMMA = CONFIG.GAMMA
        self.LR_A = CONFIG.LR_A
        self.MAX_MODEL = CONFIG.MAX_MODEL
        
        #== DQN ==
        self.device = device
        self.build_network()

    def build_network(self):
        self.Q_network = model(self.state_num, self.action_num)
        self.target_network = model(self.state_num, self.action_num)
        if self.device == torch.device('cuda'):
            self.Q_network.cuda()
            self.target_network.cuda()
        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=self.LR_A)
    
    def update_target_network(self):
        # Q_target <- Q_policy
        self.target_network.load_state_dict(self.Q_network.state_dict())
    
    def update_Q_network(self, memory):
        if len(memory) < self.BATCH_SIZE:
            return
        
        #== EXPERIENCE REPLAY ==
        transitions = memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                      device=device, dtype=torch.bool)
        non_final_state_nxt = torch.FloatTensor([s for s in batch.next_state
                                                     if s is not None])
        state = torch.FloatTensor(batch.state)
        action = torch.FloatTensor(batch.action)
        reward = torch.FloatTensor(batch.reward)
        
        state_new = torch.from_numpy(state_new).float()
        if self.device == torch.device('cuda'):
            state = state.cuda()
            action = action.cuda()
            non_final_state_nxt = non_final_state_nxt.cuda()
            reward = reward.cuda()
            
            
        #== UPDATE Q_POLICY ==
        self.Q_network.eval()
        self.target_network.eval()
        
        # get Q(s,a):
        state_action_values = self.Q_network(state).gather(1, action) # out = Q [ i ][ action[i] ]
        
        # get new action by Q_policy: a' = argmax_a' Q_policy(s', a')
        action_nxt = self.Q_network(non_final_state_nxt).max(dim=1)[1].cpu().data.view(-1, 1) # out is a column consiting of indices for max Q
        
        # get expected value: y = r + gamma * Q_tar(s', a')
        state_value_nxt = torch.zeros(self.BATCH_SIZE, device=self.device)
        Q_tmp = self.target_network(non_final_state_nxt)
        state_value_nxt[non_final_mask] = Q_tmp.gather(1, action_nxt)
        expected_state_action_values = (state_value_nxt * self.GAMMA) + reward
        
        # regression Q(s, a) -> y
        self.Q_network.train()
        loss = mse_loss(input=state_action_values, target=expected_state_action_values.detach())
        
        # backward optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.data[0]

    def take_action(self, state):
        state = torch.from_numpy(state).float()
        if self.device == torch.device('cuda'):
            state.cuda()
        
        self.Q_network.eval()
        with torch.no_grad():
            Q_s = self.Q_network.forward(state)
        val, idx = Q_s.max(dim=1)
        
        # with epsilon prob to choose random action else choose argmax Q estimate action
        if random.random() < self.epsilon:
            return random.randint(0, self.action_num-1)
        else:
            return idx.data.numpy()[0]
    '''
    def update_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon -= self.epsilon_discount_rate
    
    def stop_epsilon(self):
        self.epsilon_tmp = self.epsilon        
        self.epsilon = 0        
    
    def restore_epsilon(self):
        self.epsilon = self.epsilon_tmp        
    '''
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