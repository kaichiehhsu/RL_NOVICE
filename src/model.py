import torch
import torch.nn as nn

class model(nn.Module):
    
    def __init__(self, state_num, action_num):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=state_num, out_features=100),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU())
        self.fc3 = nn.Linear(in_features=100, out_features=action_num)
        
        self._initialize_weights()
    
    def forward(self, observation):
        out1 = self.fc1(observation)
        out2 = self.fc2(out1)  
        out  = self.fc3(out2)
        
        return out
    
    def save(self, path, step):
        torch.save({
            'step': step,
            'state_dict': self.state_dict(),
        }, path)
            
    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint['state_dict'])
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 1)
                m.bias.data.zero_()