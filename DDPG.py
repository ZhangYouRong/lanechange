# must run the server in stable FPS=20!
# for windows, enter this in the command line (in the root directory of carla)
# D:\software\CARLA_0.9.5\CarlaUE4.exe -benchmark -FPS=20

import argparse
from itertools import count

import os, sys, random
import numpy as np

from env import Env
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

'''
Implementation of Deep Deterministic Policy Gradients (DDPG) with pytorch 
riginal paper: https://arxiv.org/abs/1509.02971
Not the author's implementation !
'''

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str)  # mode = 'train' or 'test'
# Note that DDPG is feasible about hyper-parameters.
# You should fine-tuning if you change to another environment.
parser.add_argument("--env_name", default="Carla_0.9.5")
parser.add_argument('--tau', default=0.005, type=float)  # target smoothing coefficient
# parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--test_iteration', default=10, type=int)

parser.add_argument('--learning_rate', default=1e-3, type=float)
parser.add_argument('--gamma', default=0.99, type=int)  # discounted factor
parser.add_argument('--capacity', default=10000, type=int)  # replay buffer size
parser.add_argument('--batch_size', default=32, type=int)  # mini batch size
parser.add_argument('--seed', default=False, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)
# optional parameters

parser.add_argument('--sample_frequency', default=256, type=int)
parser.add_argument('--log_interval', default=50, type=int)  # saving interval of steps
parser.add_argument('--load', default=False, type=bool)  # load model (only available on test mode)
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_episode', default=10000, type=int)  # num of games
parser.add_argument('--max_length_of_time', default=35, type=int)  # num of games
parser.add_argument('--print_log', default=100, type=int)  # num of steps to print log
parser.add_argument('--update_iteration', default=10, type=int) # every step replay 10 batches for update
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# script_name = os.path.basename(__file__)
script_name=os.path.basename(os.path.splitext(__file__)[0])
env = Env()

if args.seed:
    env.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

state_dim = env.observation_space_dim
action_dim = env.action_dim
max_action = float(env.max_action)
min_Val = torch.tensor(1e-7).float().to(device)  # min value

directory = './exp'+script_name+args.env_name+'./'


class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''

    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:  # 如果满了从第一个开始刷新替换
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr+1)%self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)  # 随机选择batch size个index
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action*torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim+action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), args.learning_rate)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), args.learning_rate)
        self.replay_buffer = Replay_buffer()
        self.writer = SummaryWriter(directory)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)  # state转换为1行若干列的2维Tensor
        return self.actor(state).cpu().data.numpy().flatten()  # 调用__call__,nn.module中的__call__调用forward函数

    def update(self):

        for it in range(args.update_iteration):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(args.batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward+((1-done)*args.gamma*target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(args.tau*param.data+(1-args.tau)*target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(args.tau*param.data+(1-args.tau)*target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def save(self):
        torch.save(self.actor.state_dict(), directory+'actor.pth')
        torch.save(self.critic.state_dict(), directory+'critic.pth')
        # print("====================================")
        # print("Model has been saved...")
        # print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(directory+'actor.pth'))
        self.critic.load_state_dict(torch.load(directory+'critic.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


def main():
    ddpg_agent = DDPG(state_dim, action_dim, max_action)
    ep_r = 0
    if args.mode == 'test':
        ddpg_agent.load()
        for i in range(args.test_iteration):
            state = env.reset()
            for t in count():
                action = ddpg_agent.select_action(state)
                next_state, reward, done, info = env.step(np.float32(action))
                ep_r += reward
                # env.render()
                if done or env.simulation_time > args.max_length_of_time:
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    ep_r = 0
                    break
                state = next_state

    elif args.mode == 'train':
        print("====================================")
        print("Collection Experience...")
        print("====================================")
        if args.load:
            ddpg_agent.load()
        for i in range(args.max_episode):
            state = env.reset()
            for t in count():
                action = ddpg_agent.select_action(state)

                # issue 3 add noise to action
                action = (action+np.random.normal(0, args.exploration_noise, size=env.action_dim)).clip(
                    env.action_space_low, env.action_space_high)

                next_state, reward, done, info = env.step(action)
                ep_r += reward
                ddpg_agent.replay_buffer.push((state, next_state, action, reward, np.float(done)))

                state = next_state
                if  env.simulation_time > args.max_length_of_time:  # 一个episode结束
                    ddpg_agent.writer.add_scalar('ep_r', ep_r, global_step=i)
                    if i%args.print_log == 0:
                        print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    ep_r = 0
                    break

            if i%args.log_interval == 0:
                ddpg_agent.save()
            if len(ddpg_agent.replay_buffer.storage) >= args.capacity-1:  # 开始更新网络学习
                ddpg_agent.update()

    else:
        raise NameError("mode wrong!!!")


if __name__ == '__main__':
    main()
