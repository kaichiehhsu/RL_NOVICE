import torch
import torch.nn as nn             
                
class model(nn.Module):
    
    def __init__(self, state_num, action_num):
        super(model, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=state_num, out_features=100),
            nn.Tanh())
        self.a_head = nn.Linear(100, action_num)
        self.v_head = nn.Linear(100, 1)
        
        #self._initialize_weights()

    def forward(self, x):
        x = self.fc(x)
        a = self.a_head(x) - self.a_head(x).mean(dim=1, keepdim=True)
        v = self.v_head(x)
        action_scores = a + v
        return action_scores
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 1)
                m.bias.data.zero_()
