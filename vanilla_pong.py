import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import gym
from viewer import UniImageViewer
import cv2
import statistics
from torchvision.transforms.functional import to_tensor
import numpy as np
import threading
from tensorboardX import SummaryWriter
import random
import time
import argparse


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
            tb.add_scalar(method.__name__, (te-ts), global_step=tb_step)
        return result

    return timed


class PolicyNet(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.l1 = nn.Linear(features, 200)
        self.bn1 = nn.BatchNorm1d(200)
        self.l2 = nn.Linear(200, 1)
        self.bn2 = nn.BatchNorm1d(1)
        self.batchnorm = False

    def forward(self, observation):
        if self.batchnorm:
            hidden = torch.relu(self.bn1(self.l1(observation)))
            return torch.sigmoid(self.bn2(self.l2(hidden)))
        else:
            hidden = torch.relu(self.l1(observation))
            return torch.sigmoid(self.l2(hidden))



class RolloutDataSet(Dataset):
    def __init__(self, discount_factor):
        super().__init__()
        self.rollout = []
        self.value = []
        self.start = 0
        self.discount_factor = discount_factor

    def reset(self):
        self.rollout = []

    def add_rollout(self, rollout):
        for observation, action, reward, done, info in rollout:
            self.append(observation, reward, action, done)

    def append(self, observation, reward, action, done):
        self.rollout.append((observation, reward, action))
        if reward != 0.0:
            self.end_game()

    def end_game(self):
        values = []
        cum_value = 0.0
        # calculate values
        for step in reversed(range(self.start, len(self.rollout))):
            cum_value = self.rollout[step][1] + cum_value * self.discount_factor
            values.append(cum_value)
        self.value = self.value + list(reversed(values))
        self.start = len(self.rollout)

    def normalize(self):
        mean = statistics.mean(self.value)
        stdev = statistics.stdev(self.value)
        self.value = [(vl - mean) / stdev for vl in self.value]

    def total_reward(self):
        return sum([reward[1] for reward in self.rollout])

    def __getitem__(self, item):
        observation, reward, action = self.rollout[item]
        try:
            value = self.value[item]
        except IndexError:
            print(f'missing value at index {item} reward{reward}')
            value = 0

        observation_t = to_tensor(np.expand_dims(observation, axis=2))
        return observation_t, reward, action, value

        reward_t = torch.tensor([reward], dtype=torch.float32)
        action_t = torch.tensor([action], dtype=torch.long)
        value_t = torch.tensor([value], dtype=torch.float32)
        return observation_t, reward_t, action_t, value_t

    def __len__(self):
        return len(self.rollout)


def downsample(observation):
    greyscale = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    greyscale = cv2.resize(greyscale, downsample_image_size, cv2.INTER_LINEAR)
    return greyscale


@timeit
def rollout_policy(policy):
    global tb_step, rundir
    policy = policy.eval()
    policy = policy.to(torch.device('cpu'))
    rollout_dataset = RolloutDataSet(discount_factor=0.99)
    reward_total = 0

    for i in range(num_rollouts):

        game_length = 0
        gl = []

        observation_t0 = env.reset()
        observation_t0 = downsample(observation_t0)
        action = default_action
        observation_t1, reward, done, info = env.step(action)
        observation_t1 = downsample(observation_t1)
        observation = observation_t1 - observation_t0
        observation_t0 = observation_t1
        done = False

        while not done:
            # take an action on current observation and record result
            observation_tensor = to_tensor(np.expand_dims(observation, axis=2)).squeeze().unsqueeze(0).view(-1,
                                                                                                            features)
            action_prob = policy(observation_tensor)
            action = 2 if np.random.uniform() < action_prob.item() else 3
            observation_t1, reward, done, info = env.step(action)
            reward_total += reward

            rollout_dataset.append(observation, reward, action, done)

            # compute the observation that resulted from our action
            observation_t1 = downsample(observation_t1)
            observation = observation_t1 - observation_t0
            observation_t0 = observation_t1

            if reward == 0:
                game_length += 1
            else:
                gl.append(game_length)
                game_length = 0

            if i == 0 and epoch % 10 == 0 and view_games:
                v.render(observation)
                env.render(mode='human')

        # hooks.execute_test_end(collected_rollouts, num_rollouts, reward_total)
        tb.add_scalar('reward', reward_total, tb_step)
        tb.add_scalar('ave_game_len', statistics.mean(gl), tb_step)
        reward_total = 0
        tb_step += 1

    if epoch % 20 == 0:
        torch.save(policy.state_dict(), rundir + '/vanilla.wgt')

    rollout_dataset.normalize()
    return rollout_dataset


@timeit
def train_policy(policy, rollout_dataset, optim, device, batch_size, num_workers):

        policy = policy.train()
        policy = policy.to(torch.device(device))
        pin_memory = False if device is 'cpu' else True
        batch_size = batch_size if batch_size != 0 else len(rollout_dataset)

        rollout_loader = DataLoader(rollout_dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers)
        for i, (observation, reward, action, value) in enumerate(rollout_loader):
            observation = observation.to(device)
            action = action.to(device).float()
            value = value.to(device).float()
            action[action == 2] = 1.0
            action[action == 3] = 0.0
            optim.zero_grad()
            action_prob = policy(observation.squeeze().view(-1, features)).squeeze()
            prob_action_taken = action * torch.log(action_prob + 1e-12) + (1.0 - action) * torch.log(
                (1.0 - action_prob + 1e-12))
            loss = - value * prob_action_taken
            loss = loss.sum()
            loss.backward()
            optim.step()
            # hooks.execute_train_end(i, len(rollout_loader), loss.item())


if __name__ == '__main__':

    parser = argparse.ArgumentParser('learn to play pong')
    parser.add_argument('--reload')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--batchnorm', dest='batchnorm', action='store_true')
    parser.set_defaults(batchnorm=False)
    args = parser.parse_args()

    max_rollout_len = 3000
    downsample_image_size = (100, 80)
    features = downsample_image_size[0] * downsample_image_size[1]
    default_action = 2
    rundir = f'runs/rmsprop_{random.randint(0,100)}'
    tb = SummaryWriter(rundir)
    tb_step = 0
    num_epochs = 600
    num_rollouts = 10
    collected_rollouts = 0
    view_games = False

    env = gym.make('Pong-v0')
    v = UniImageViewer('pong', (200, 160))
    # GUIProgressMeter('training_pong')
    policy_net = PolicyNet(features)
    if args.reload:
        policy_net.load_state_dict(torch.load(args.reload))
    policy_net.batchnorm = args.batchnorm

    optim = torch.optim.RMSprop(lr=1e-3, params=policy_net.parameters())

    for epoch in range(num_epochs):
        rollout_dataset = rollout_policy(policy_net)
        tb.add_scalar('collected_frames', len(rollout_dataset), tb_step)
        train_policy(policy_net, rollout_dataset, optim, args.device, args.batch_size, args.workers)

    # hooks.execute_epoch_end(epoch, num_epochs)
