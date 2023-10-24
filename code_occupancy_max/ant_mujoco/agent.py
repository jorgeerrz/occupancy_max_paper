import numpy as np
import scipy.signal
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import time
import os.path as osp

from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time

from logx import EpochLogger, setup_logger_kwargs

from models import *
from memory import *
from utils import *


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MOP:
    def __init__(self, exp_name, env_fn, actor_critic=MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=16, num_test_episodes=5, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1) -> None:
        
        self.exp_name = exp_name
        self.gamma = gamma
        self.alpha = alpha
        self.env_fn = env_fn
        self.steps_per_epoch=steps_per_epoch
        self.epochs=epochs
        self.polyak=polyak
        self.batch_size=batch_size
        self.start_steps=start_steps 
        self.update_after=update_after
        self.update_every=update_every
        self.num_test_episodes=num_test_episodes
        self.max_ep_len=max_ep_len
        self.save_freq = save_freq
        self.seed = seed
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.env, self.test_env = self.env_fn(), self.env_fn()
        obs_dim = self.env.observation_space.shape
        act_dim = self.env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = self.env.action_space.high[0]

        
        # Create actor-critic module and target networks
        self.ac = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs).to(device)
        self.ac_targ = deepcopy(self.ac).to(device)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
            
        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        self.var_counts = tuple(count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d \n' %self.var_counts)
        
        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=lr)
        self.q_optimizer = Adam(self.q_params, lr=lr)

        # Set up model saving
        self.logger.setup_pytorch_saver(self.ac)
        
            
    # Set up function for computing MOP Q-losses
    def update_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)
                    
        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)
                
        
        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.cpu().detach().numpy(), Q2Vals=q2.cpu().detach().numpy())

        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()
        
        # Record things
        self.logger.store(LossQ=loss_q.item(), **q_info)

    # Set up function for computing MOP pi loss
    def update_pi(self, data):
        o= data['obs']
        
        
        pi, logp_pi, std = self.ac.pi(o, with_std=True)
        
        self.logger.store(std=std.mean().item())
        
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()
            

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.cpu().detach().numpy())

        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()
        
        # Record things
        self.logger.store(LossPi=loss_pi.item(), **pi_info)
        
            
    def update(self, data):
        # First run one gradient descent step for Q1 and Q2 
        self.update_q(data)

        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.update_pi(data)

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
            
        

    def get_action(self, o, deterministic=False):
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32).to(device), deterministic)

    def test_agent(self):
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset()[0], False, 0, 0
            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ , info = self.test_env.step(self.get_action(o, deterministic=True))
                r = self.reward_function(r)
                ep_ret += r
                ep_len += 1
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    
    def reward_function(self, reward):
        return reward
        
        
    def train(self):
        # Prepare for interaction with environment
        total_steps = self.steps_per_epoch * self.epochs
        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset()[0], 0, 0

        # Main loop: collect experience in env and update/log each epoch
        for t in range(total_steps):
            
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy. 
            if t > self.start_steps:
                a = self.get_action(o, deterministic=False)
            else:
                a = self.env.action_space.sample()

            # Step the env
            o2, r, d, _, _ = self.env.step(a)
            r = self.reward_function(r)
            ep_ret += r
            ep_len += 1

            d = False if ep_len==self.max_ep_len else d

            # Store experience to replay buffer
            self.replay_buffer.store(o, a, r, o2, d)
            
            o = o2

            # End of trajectory handling
            if d or (ep_len == self.max_ep_len):
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = self.env.reset()[0], 0, 0

            # Update handling
            if t >= self.update_after and t % self.update_every == 0:
                for j in range(self.update_every):
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    self.update(data=batch)

            # End of epoch handling
            if (t+1) % self.steps_per_epoch == 0:
                epoch = (t+1) // self.steps_per_epoch

                # Save model
                if (epoch % self.save_freq == 0) or (epoch == self.epochs):
                    self.logger.save_state({'env': self.env}, None)

                # Test the performance of the deterministic version of the agent.
                self.test_agent()

                if t < self.update_after: continue
                 
                # Log info about epoch
                self.logger.log_tabular('Experiment Name', self.exp_name + "_" +str(self.seed))
                self.logger.log_tabular('Epoch', epoch)
                self.logger.log_tabular('EpRet', average_only=True)
                self.logger.log_tabular('TestEpRet', average_only=True)
                self.logger.log_tabular('EpLen', average_only=True)
                self.logger.log_tabular('TestEpLen', average_only=True)
                self.logger.log_tabular('TotalEnvInteracts', t)
                self.logger.log_tabular('Q1Vals', average_only=True)
                self.logger.log_tabular('Q2Vals', average_only=True)
                self.logger.log_tabular('LogPi', average_only=True)
                self.logger.log_tabular('LossPi', average_only=True)
                self.logger.log_tabular('LossQ', average_only=True)
                self.logger.log_tabular('std', average_only=True)
                self.logger.log_tabular('Time', time.time()-start_time)
                self.logger.dump_tabular()



class EGready(MOP):
    def __init__(self, exp_name, env_fn, actor_critic=MLPActorCritic, epsilon=2, ac_kwargs=dict(), **kwargs) -> None:
        
        super().__init__(exp_name, env_fn, actor_critic, ac_kwargs, **kwargs)
        self.exp_name = exp_name
        self.epsilon = epsilon
            
    def get_action(self, o, **kwargs):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.ac.act(torch.as_tensor(o, dtype=torch.float32).to(device), deterministic=True)

        
