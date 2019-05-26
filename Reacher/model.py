import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)
class Actor(nn.Module):
    def __init__(self,state_size,action_size):
        super(Actor, self).__init__()
        self.state_size=state_size
        self.action_size=action_size
        self.seed = torch.manual_seed(7)
        self.fc1=nn.Linear(self.state_size,400)
        self.bn1 = nn.BatchNorm1d(400)
        self.fc2=nn.Linear(400,300)
        self.fc3=nn.Linear(300,self.action_size)
        self.reset_parameters()
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3,3e-3)
    def forward(self,state):
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))
class Critic(nn.Module):
    def __init__(self,state_size,action_size):
        super(Critic,self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(7)
        self.fc1 = nn.Linear(self.state_size,400)
        self.bn1 = nn.BatchNorm1d(400)
        self.fc2 = nn.Linear(400+self.action_size,300)
        self.fc3 = nn.Linear(300,1)
        self.reset_parameters()
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3,3e-3)
    def forward(self,state,action):
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.fc2(torch.cat((x,action),dim=1)))
        return self.fc3(x)