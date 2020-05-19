from collections import namedtuple
import random

Transition = namedtuple('Transition', ['s', 'a', 'r', 's_'])

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.isfull = False

    def update(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = int((self.position + 1) % self.capacity)
        if len(self.memory) == self.capacity:
            self.isfull = True

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)