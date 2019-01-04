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
from storm.vis.hooks import hooks
from storm.vis import GUIProgressMeter, TB
from tensorboardX import SummaryWriter
from multithread import GymWorker

max_rollout_len = 3000
downsample_image_size = (100, 80)
features = downsample_image_size[0] * downsample_image_size[1]
tb = SummaryWriter(f'runs/rmsprop_1e3_nobatch_multi_double_3')
tb_step = 0
num_epochs = 600
num_rollouts = 10
collected_rollouts = 0
resume = False
save = True
view_games = False


env = gym.make('Pong-v0')
#v = UniImageViewer('pong', (200, 160))
#GUIProgressMeter('training_pong')


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
    def __init__(self, discount_factor, num_rollouts):
        super().__init__()
        self.rollout = []
        self.value = []
        self.start = 0
        self.discount_factor = discount_factor
        self.lock = threading.Lock()
        self.game_length = 0
        self.gl = []
        self.collected_rollouts = 0
        self.num_rollouts = num_rollouts
        self.reward_total = 0

    def reset(self):
        self.rollout = []

    def add_rollout(self, rollout):
        """Threadsafe method to add rollouts to the dataset"""
        global tb_step
        with self.lock:
            for observation, action, reward, done, info in rollout:
                self.append(observation, reward, action, done)

                if reward == 0:
                    self.game_length += 1
                else:
                    self.gl.append(self.game_length)
                    self.reward_total += reward
                    self.game_length = 0

                if done:
                    self.collected_rollouts += 1
                    #hooks.execute_test_end(self.collected_rollouts, num_rollouts, self.reward_total)
                    tb.add_scalar('reward', self.reward_total, tb_step)
                    tb.add_scalar('ave_game_len', statistics.mean(self.gl), tb_step)
                    self.gl = []
                    self.reward_total = 0
                    tb_step += 1

    def append(self, observation, reward, action, done):
        self.rollout.append((observation, reward, action, done))
        if reward != 0.0:
            self.end_game()
            print(f'game finished reward: {reward}', ' !!!!!' if reward == 1.0 else '')

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
        observation, reward, action, done = self.rollout[item]
        value = self.value[item]
        observation_t = to_tensor(np.expand_dims(observation, axis=2)).double()
        return observation_t, reward, action, value, done

    def __len__(self):
        return len(self.rollout)


def downsample(observation):
    greyscale = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    greyscale = cv2.resize(greyscale, downsample_image_size, cv2.INTER_LINEAR)
    return greyscale


policy_net = PolicyNet(features).double()
if resume:
    policy_net.load_state_dict(torch.load('vanilla.wgt'))

#optim = torch.optim.Adam(lr=1e-3, params=policy_net.parameters())
optim = torch.optim.RMSprop(lr=1e-3, params=policy_net.parameters())
#optim = torch.optim.SGD(lr=3e3, params=policy_net.parameters())

for epoch in range(num_epochs):
    rollout = RolloutDataSet(discount_factor=0.99, num_rollouts=num_rollouts)
    threads = []

    # Create new threads
    for id in range(1, 6):
        threads.append(GymWorker(id, f"Thread-{id}", id, 'Pong-v0', rollout, 2, 2, policy_net, features,
                                 preprocessor=downsample))

    # Start new Threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    if rollout.collected_rollouts == 0 and epoch % 10 == 0 and view_games:
        v = UniImageViewer('match replay')
        done = False
        for observation_t, reward, action, value, done in iter(rollout):
            if done:
                break
            v.render(observation_t)

    if epoch % 20 == 0 and save:
        torch.save(policy_net.state_dict(), 'vanilla.wgt')

    policy_net = policy_net.train()
    rollout.normalize()
    rollout_loader = DataLoader(rollout, batch_size=len(rollout))

    print('training')

    for i, (observation, reward, action, value, done) in enumerate(rollout_loader):
        action = action.double()
        value = value.double()
        action[action == 2] = 1.0
        action[action == 3] = 0.0
        optim.zero_grad()
        action_prob = policy_net(observation.squeeze().view(-1, features)).squeeze()
        dp = action - action_prob
        prob_action_taken = action * torch.log(action_prob + 1e-12) + (1 - action) * (torch.log(1 - action_prob))
        #prob_action_taken = action * action_prob + (1.0 - action) * (1.0 - action_prob)
        loss = - value * prob_action_taken
        loss = loss.sum()
        loss.backward()
        optim.step()
        #hooks.execute_train_end(i, len(rollout_loader), loss.item())

    #hooks.execute_epoch_end(epoch, num_epochs)