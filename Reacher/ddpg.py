from collections import deque,namedtuple
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class ReplayBuffer():
    def __init__(self,max_size):
        self.buffer = deque(maxlen=max_size)
        self.exp = namedtuple('experinece',field_names=['state','action','reward','next_state','done'])
        self.seed = random.seed(7)
    def __len__(self):
        return len(self.buffer)
    def Add(self,state,action,reward,next_state,done):
        e = self.exp(state=state,action=action,reward=reward,next_state=next_state,done=done)
        self.buffer.append(e)
    def sample(self,batch_size):
        index = random.sample(self.buffer,batch_size)
        state=torch.from_numpy(np.vstack([i.state for i in index if i is not None])).to(torch.float32).to(device)
        action=torch.from_numpy(np.vstack([i.action for i in index if i is not None])).to(torch.float32).to(device)
        reward=torch.from_numpy(np.vstack([i.reward for i in index if i is not None])).to(torch.float32).to(device)
        next_state=torch.from_numpy(np.vstack([i.next_state for i in index if i is not None])).to(torch.float32).to(device)
        done=torch.from_numpy(np.vstack([i.done for i in index if i is not None]).astype(np.uint8)).to(torch.float32).to(device)
        return (state,action,reward,next_state,done)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size,  mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(7)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state

class DDPGAgent():
    def __init__(self,batch_size,state_size,action_size,replaybuffer):
        self.batch_size=batch_size
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer = replaybuffer
        self.seed = random.seed(7)
        self.actor_local = Actor(self.state_size,action_size).to(device)
        self.critic_local = Critic(self.state_size,self.action_size).to(device)
        
        self.actor_target = Actor(self.state_size,self.action_size).to(device)
        self.critic_target = Critic(self.state_size,self.action_size).to(device)
        self.lr_actor=1e-3
        self.lr_critic=1e-3
        self.optim_actor = optim.Adam(self.actor_local.parameters(),self.lr_actor)
        self.optim_critic = optim.Adam(self.critic_local.parameters(),self.lr_critic,weight_decay=0.00)
        
        self.noise = OUNoise(self.action_size)
        self.gamma =0.99
    def reset(self):
        self.noise.reset()
    def update_parameters(self,local,target,tau):
        for lp,tp in zip(local.parameters(),target.parameters()):
            tp.data.copy_(lp.data*tau+tp.data*(1-tau))
    def policy(self,state):
        state = torch.from_numpy(state).to(torch.float32).to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        action = action+self.noise.sample()
        return np.clip(action,-1,1)
            
    def learn(self):
        state,action,reward,next_state,done = self.replay_buffer.sample(self.batch_size)
        
        action_next = self.actor_target(next_state)
        qvalue_next = self.critic_target(next_state,action_next)
        Qtarget = reward + (self.gamma*qvalue_next*(1-done))
        
        Qcurrent = self.critic_local(state,action)
        
        critic_loss = F.mse_loss(Qcurrent,Qtarget)
        
        self.optim_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.optim_critic.step()
        
        action_c = self.actor_local(state)
        actor_loss = -self.critic_local(state,action_c).mean()
        
        self.optim_actor.zero_grad()
        actor_loss.backward()
        self.optim_actor.step()
        
        self.update_parameters(self.actor_local,self.actor_target,1e-3)
        self.update_parameters(self.critic_local,self.critic_target,1e-3)
    def adjust_learning_rate(self,optimizer,decay_rate,lr):
        # sets the learning rate to the initial LR decayed by 0.1 every 'each' iterations
        lr = lr*decay_rate 
        state_dict = optimizer.state_dict()
        for param_group in state_dict['param_groups']:
            param_group['lr'] = lr
        optimizer.load_state_dict(state_dict)
        return lr
    def update_lr(self,decay=0.99):
        self.lr_actor=self.adjust_learning_rate(self.optim_actor,decay,self.lr_actor)
        self.lr_critic=self.adjust_learning_rate(self.optim_critic,decay,self.lr_critic)