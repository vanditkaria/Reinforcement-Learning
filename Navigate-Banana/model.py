import torch
import torch.nn as nn
import torch.nn.functional as F

class Q_Network(nn.Module):
    '''
    Used for getting the action value for given state
    '''
    def __init__(self, state_size, action_size):
        '''
        initlize layer to get output similar to action size
        '''
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc_1 = nn.Linear(self.state_size, 256)
        self.fc_2 = nn.Linear(256, 128)
        self.fc_3 = nn.Linear(128, 64)
        self.ad = nn.Linear(64, self.action_size) # advantage estimate for given state and action pair
        self.va = nn.Linear(64, 1) # value estimate for given state
    def __call__(self,x):
        '''
        return the action value given the state
        Input
        x state representation
        Return
        action value
        '''
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        va = self.va(x)
        ad = self.ad(x)
        q = va + ad - (torch.mean(ad, 1, keepdim=True)) # a(s,a) = q(s,a) - v(s)
        return q
