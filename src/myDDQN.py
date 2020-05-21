import argparse
import pickle
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import random

import gym
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, smooth_l1_loss
import torch.optim as optim

from config import config
from DDQN import DDQN

parser = argparse.ArgumentParser(description='Solve the Pendulum-v0 with DQN')
parser.add_argument('--gamma', type=float, default=0.9, metavar='G', help='discount factor (default: 0.9)')
parser.add_argument('--num_actions', type=int, default=5, metavar='N', help='discretize action space (default: 5)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('-r','--render', action='store_true', help='render the environment')
parser.add_argument('--log-interval',type=int,default=10,metavar='N',help='interval between training status logs (default: 10)')
parser.add_argument('-a','--act_num',type=int,default=5,metavar='N',help='the number of possible actions')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

TrainingRecord = namedtuple('TrainingRecord', ['ep', 'reward'])

def main():
    #== ENVIRONMENT ==
    env = gym.make('Pendulum-v0')
    env.seed(args.seed)
    
    #== CONFIGURATION ==
    print('='*4 + ' CONFIGURATION ' + '='*4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    CONFIG = config(RENDER=True, MAX_EPISODES=200, MAX_EP_STEPS=200, TARGET_UPDATE=10)
    for key, value in CONFIG.__dict__.items():
        if key[:1] != '_': print(key, value) 
    
    #== AGENT ==
    agent=DDQN(3, args.act_num, device, CONFIG)
    
    #== TRAINING ==
    training_records = []
    running_reward = -1000
    for i_ep in range(CONFIG.MAX_EPISODES):
        ep_reward = 0
        state = env.reset()

        for t in range(CONFIG.MAX_EP_STEPS):
            action, action_index = agent.select_action(state)
            state_, reward, done, _ = env.step(action)
            ep_reward += reward
            if args.render:
                env.render()
                
            # reward \in [~-16, 0] -> shift to about [-1, 1]
            agent.store_transition(state, action_index, (reward+8)/8, state_)
            state = state_
            q = agent.update()

        running_reward = running_reward * 0.9 + ep_reward * 0.1
        training_records.append(TrainingRecord(i_ep, running_reward))

        if i_ep % args.log_interval == 0:
            print('Ep[{:3.0f}]: Running Reward: {:.2f} \t Real Reward: {:.2f}'.format(i_ep, running_reward, ep_reward))
        if running_reward > -180:
            print("Solved! Running reward is now {:.2f}!".format(running_reward))
            env.close()
            break

    env.close()

    plt.plot([r.ep for r in training_records], [r.reward for r in training_records])
    plt.title('DQN')
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.savefig("img/dqn.png")
    plt.show()


if __name__ == '__main__':
    main()