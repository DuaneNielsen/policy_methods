from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['GPU_DEBUG']='0'

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
            if 'tb' in globals():
                tb.add_scalar(method.__name__, (te - ts), global_step=tb_step)
        return result

    return timed


class PolicyNet(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.l1 = nn.Linear(features, 200)
        self.bn1 = nn.BatchNorm1d(200)
        self.l2 = nn.Linear(200, 1)
        self.bn2 = nn.BatchNorm1d(1)

    def forward(self, observation):
        hidden = torch.relu(self.l1(observation))
        return torch.sigmoid(self.l2(hidden))


class RolloutDataSet(Dataset):
    def __init__(self, discount_factor):
        super().__init__()
        self.rollout = []
        self.value = []
        self.start = 0
        self.discount_factor = discount_factor
        self.lock = threading.Lock()

    def reset(self):
        self.rollout = []

    def add_rollout(self, rollout):
        self.lock.acquire()
        for observation, action, reward, done, info in rollout:
            self.append(observation, reward, action, done)
        self.lock.release()

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
        value = self.value[item]
        observation_t = to_tensor(np.expand_dims(observation, axis=2))
        return observation_t, reward, action, value

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
    policy = policy.to('cpu')
    rollout_dataset = RolloutDataSet(discount_factor=0.99)
    reward_total = 0

    for i in range(num_rollouts):

        game_length = 0
        gl = []
        probs = []

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
            observation_tensor = to_tensor(np.expand_dims(observation, axis=2))\
                .squeeze().unsqueeze(0).view(-1, policy.features)
            action_prob = policy(observation_tensor)
            action = 2 if np.random.uniform() < action_prob.item() else 3
            observation_t1, reward, done, info = env.step(action)
            reward_total += reward

            rollout_dataset.append(observation, reward, action, done)

            # compute the observation that resulted from our action
            observation_t1 = downsample(observation_t1)
            observation = observation_t1 - observation_t0
            observation_t0 = observation_t1

            # monitoring
            if reward == 0:
                game_length += 1
                if debug:
                    probs.append(torch.exp(action_prob.squeeze()))
            else:
                gl.append(game_length)
                game_length = 0

                if debug:
                    probs = torch.stack(probs)
                    mean = probs.mean(dim=0)
                    print(mean[0].item(), mean[1].item())
                    del probs
                    probs = []

            if view_games:
                v.render(observation)
                env.render(mode='human')

        # more monitoring
        # hooks.execute_test_end(collected_rollouts, num_rollouts, reward_total)
        tb.add_scalar('reward', reward_total, tb_step)
        tb.add_scalar('ave_game_len', statistics.mean(gl), tb_step)
        reward_total = 0
        tb_step += 1

    # save the file every so often
    if epoch % 20 == 0:
        torch.save(policy.state_dict(), rundir + '/vanilla.wgt')

    rollout_dataset.normalize()
    return rollout_dataset


def ppo_loss(newprob, oldprob, advantage, clip=0.2):

    ratio = newprob / oldprob

    clipped_ratio = ratio.clamp(1.0 - clip, 1.0 + clip)
    clipped_step = clipped_ratio * advantage
    full_step = ratio * advantage
    min_step = torch.stack((full_step, clipped_step), dim=1)
    min_step, clipped = torch.min(min_step, dim=1)

    if debug:
        print(f'ADVTG {advantage[0].data}')
        print(f'NEW_P {newprob[0].data}')
        print(f'OLD_P {oldprob[0].data}')
        print(f'RATIO {ratio[0].data}')
        print(f'CLIP_ {clipped_step[0].data}')

    min_step *= -1
    return min_step.mean()


@timeit
def train_policy(policy, rollout_dataset, optim, device='cpu'):
    policy = policy.train()
    policy = policy.to(device)

    batches = math.floor(len(rollout_dataset) / max_minibatch_size) + 1
    batch_size = math.floor(len(rollout_dataset) / batches)
    steps_per_batch = math.floor(12 / batches) if math.floor(12/batches) > 0 else 1
    tb.add_scalar('batches', batches, tb_step)

    rollout_loader = DataLoader(rollout_dataset, batch_size=batch_size, shuffle=True)
    batches_p = 0
    for i, (observation, reward, action, advantage) in enumerate(rollout_loader):
        batches_p += 1
        for step in range(steps_per_batch):

            observation = observation.to(device)
            advantage = advantage.float().to(device)
            action = action.squeeze().to(device)
            optim.zero_grad()

            if debug:
                print(f'ACT__ {action[0].data}')

            new_logprob = policy(observation.squeeze().view(-1, policy.features)).squeeze()
            new_prob = torch.exp(torch.distributions.Categorical(logits=new_logprob).log_prob(action))
            new_logprob.retain_grad()
            old_logprob = policy(observation.squeeze().view(-1, policy.features), old=True).squeeze()
            old_prob = torch.exp(torch.distributions.Categorical(logits=old_logprob).log_prob(action))
            policy.backup()

            loss = ppo_loss(new_prob, old_prob, advantage, clip=0.2)
            loss.backward()
            optim.step()
            updated_logprob = policy(observation.squeeze().view(-1, features)).squeeze()

            if debug:
                print(f'CHNGE {( torch.exp(updated_logprob) - torch.exp(new_logprob) ).data[0]}')
                print(f'NEW_G {torch.exp(new_logprob.grad.data[0])}')

            tb.add_scalar('memory_allocated', torch.cuda.memory_allocated(), tb_step)
            tb.add_scalar('memory_cached', torch.cuda.memory_cached(), tb_step)
    print(f'processed {batches_p} batches')
    #gpu_profile(frame=sys._getframe(), event='line', arg=None)


if __name__ == '__main__':

    #from gpu_memory_profiling import gpu_profile
    #import sys
    #sys.settrace(gpu_profile)

    max_rollout_len = 3000
    downsample_image_size = (100, 80)
    features = downsample_image_size[0] * downsample_image_size[1]
    default_action = 2
    rundir = f'runs/rmsprop_{random.randint(0,100)}'
    tb = SummaryWriter(rundir)
    tb_step = 0
    num_epochs = 6000
    num_rollouts = 10
    collected_rollouts = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_minibatch_size = 40000
    resume = False
    view_games = False
    debug = False

    env = gym.make('Pong-v0')
    v = UniImageViewer('pong', (200, 160))
    # GUIProgressMeter('training_pong')
    pong_action_map = [0, 2, 3]
    policy_net = PPOWrap(features, pong_action_map)
    if resume:
        policy_net.load_state_dict(torch.load('runs/ppo_multilabel_259/vanilla.wgt'))

    optim = torch.optim.Adam(lr=1e-4, params=policy_net.new.parameters())

    for epoch in range(num_epochs):
        rollout_dataset = rollout_policy(policy_net)
        tb.add_scalar('collected_frames', len(rollout_dataset), tb_step)
        train_policy(policy_net, rollout_dataset, optim, device)
        torch.cuda.empty_cache()
        #gpu_profile(frame=sys._getframe(), event='line', arg=None)

    # hooks.execute_epoch_end(epoch, num_epochs)
