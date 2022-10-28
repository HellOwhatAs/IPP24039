from typing import List
"""
Reference: https://mofanpy.com/tutorials/machine-learning/torch/DQN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

if __name__=="__main__":
    from __init__ import single_host
else:
    from stragety import single_host

class Net(nn.Module):
    def __init__(self, 
        N_ACTIONS,
        N_STATES
    ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class DQN(object):
    def __init__(self,
        N_ACTIONS:int,
        N_STATES:int,
        ENV_A_SHAPE:int,
        BATCH_SIZE = 32,
        LR = 0.01,                  # learning rate
        EPSILON = 0.9,              # greedy policy
        GAMMA = 0.9,                # reward discount
        TARGET_REPLACE_ITER = 100,   # target update frequency
        MEMORY_CAPACITY = 2000
    ):
        self.BATCH_SIZE=BATCH_SIZE
        self.LR=LR
        self.EPSILON=EPSILON
        self.GAMMA=GAMMA
        self.TARGET_REPLACE_ITER=TARGET_REPLACE_ITER
        self.MEMORY_CAPACITY=MEMORY_CAPACITY
        self.N_ACTIONS=N_ACTIONS
        self.N_STATES=N_STATES
        self.ENV_A_SHAPE=ENV_A_SHAPE

        self.eval_net, self.target_net = Net(self.N_ACTIONS,self.N_STATES), Net(self.N_ACTIONS,self.N_STATES)

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < self.EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if self.ENV_A_SHAPE == 0 else action.reshape(self.ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, self.N_ACTIONS)
            action = action if self.ENV_A_SHAPE == 0 else action.reshape(self.ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.N_STATES])
        b_a = torch.LongTensor(b_memory[:, self.N_STATES:self.N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.N_STATES+1:self.N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + self.GAMMA * q_next.max(1)[0].view(self.BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class DQN_Manager:
    def __init__(self,dqn:DQN):
        self.dqn=dqn
        self.multi_hosts_s:np.ndarray=None
        self.multi_hosts_new_s:np.ndarray=None
        self.multi_hosts_a:int=None
    @staticmethod
    def make_state(
            cores_state:    List[List[float]], 
            arrive_time:    float, 
            task_locations: List[List[int]], 
            task_size:      List[float],
            cost:           float
        )->np.ndarray:
        return np.maximum(np.array(cores_state).flatten()-arrive_time,0)
    def multi_hosts(
            self,
            cores_state:    List[List[float]], 
            arrive_time:    float, 
            task_locations: List[List[int]], 
            task_size:      List[float],
            cost:           float
        )->int:
        self.multi_hosts_new_s = self.make_state(
            cores_state,
            arrive_time,
            task_locations,
            task_size,
            cost
        )
        self.multi_hosts_a = self.dqn.choose_action(self.multi_hosts_new_s)
        return self.multi_hosts_a

    def learn(self,r:float):
        if self.multi_hosts_s is None:
            self.multi_hosts_s = self.multi_hosts_new_s
            self.multi_hosts_new_s = None
            return
        self.dqn.store_transition(
            self.multi_hosts_s,
            self.multi_hosts_a,
            r,
            self.multi_hosts_new_s
        )
        self.multi_hosts_s = self.multi_hosts_new_s
        self.multi_hosts_new_s = None

        if self.dqn.memory_counter > self.dqn.MEMORY_CAPACITY:
            self.dqn.learn()