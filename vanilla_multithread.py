import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import gym
import cv2
import statistics
from torchvision.transforms.functional import to_tensor
import numpy as np
from torch.multiprocessing import Queue, Process, Manager
from tensorboardX import SummaryWriter
import cProfile
import time
import random
import sys

def do_cprofile(func):
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats()
            import pstats
            p = pstats.Stats('output.prof')
            p.strip_dirs().sort_stats(-1).print_stats()
    return profiled_func


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
        self.frames = []
        self.value = []
        self.start = 0
        self.discount_factor = discount_factor
        self.game_length = 0
        self.gl = []
        self.collected_rollouts = 0
        self.reward_total = 0
        self.rollouts = Manager().Queue()

    def add_rollout(self, rollout):
        """Threadsafe method to add rollouts to the dataset"""
        self.rollouts.put(rollout)

    def post_process(self):
        """Call to process data after all data collected"""
        while not self.rollouts.empty():
            self.process_rollout(self.rollouts.get())
        self.normalize()

    def process_rollout(self, rollout):
        global tb_step
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
                tb.add_scalar('reward', self.reward_total, tb_step)
                tb.add_scalar('ave_game_len', statistics.mean(self.gl), tb_step)
                self.gl = []
                self.reward_total = 0
                tb_step += 1

    def append(self, observation, reward, action, done):
        self.frames.append((observation, reward, action, done))
        if reward != 0.0:
            self.end_game()

    def end_game(self):
        values = []
        cum_value = 0.0
        # calculate values
        for step in reversed(range(self.start, len(self.frames))):
            cum_value = self.frames[step][1] + cum_value * self.discount_factor
            values.append(cum_value)
        self.value = self.value + list(reversed(values))
        self.start = len(self.frames)

    def normalize(self):
        mean = statistics.mean(self.value)
        stdev = statistics.stdev(self.value)
        self.value = [(vl - mean) / stdev for vl in self.value]

    def total_reward(self):
        return sum([reward[1] for reward in self.frames])

    def __getitem__(self, item):
        observation, reward, action, done = self.frames[item]
        value = self.value[item]
        observation_t = to_tensor(np.expand_dims(observation, axis=2))
        return observation_t, reward, action, value, done

    def __len__(self):
        return len(self.frames)


class GymWorker(Process):
    def __init__(self, threadID, name, counter, env, experience_buffer, num_rollouts, initial_action,
                 policy, features, device, downsample_image_size):
        Process.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.env = env
        self.num_rollouts = num_rollouts
        self.buffer = []
        self.experience_buffer = experience_buffer
        self.downsample_image_size = downsample_image_size
        self.default_action = initial_action
        self.policy = policy.eval()
        self.features = features
        self.device = device

    def pre_process(self, observation):
        greyscale = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        greyscale = cv2.resize(greyscale, self.downsample_image_size, cv2.INTER_LINEAR)
        return greyscale

    def run(self):
        try:
            print(f'Thread {self.name}: starting')
            for i in range(self.num_rollouts):
                observation_t0 = self.env.reset()
                observation_t0 = self.pre_process(observation_t0)
                observation_t1, reward, done, info = self.env.step(self.default_action)
                observation_t1 = self.pre_process(observation_t1)
                observation = observation_t1 - observation_t0
                observation_t0 = observation_t1
                done = False
                while not done:
                    obs_t = to_tensor(np.expand_dims(observation, axis=2)).squeeze().unsqueeze(0).view(-1, self.features).to(self.device)
                    action_prob = self.policy(obs_t)
                    action = 2 if np.random.uniform() < action_prob.item() else 3
                    observation_t1, reward, done, info = self.env.step(action)

                    # save here
                    self.buffer.append((observation, action, reward, done, info))

                    # next frame
                    observation_t1 = self.pre_process(observation_t1)
                    observation = observation_t1 - observation_t0
                    observation_t0 = observation_t1

                print(f'Thread {self.name}: adding rollout')
                self.experience_buffer.add_rollout(self.buffer)
                self.buffer = []
                print(f'Thread {self.name}: added rollout')
            #self.experience_buffer.rollouts.close()
            #self.experience_buffer.rollouts.cancel_join_thread()
            print(self.experience_buffer.rollouts.qsize())
            print(f'Thread {self.name}: exiting')
        except:
            print('exception ' + sys.exc_info()[0])



@timeit
def collect_rollouts(policy_net, envs, features, num_threads, num_rollouts):
    rollout_dataset = RolloutDataSet(discount_factor=0.99)
    threads = []
    policy_net = policy_net.to(torch.device('cpu'))
    # Create new threads
    for id in range(num_threads):
        threads.append(GymWorker(id, f"Thread-{id}", id, envs[id], rollout_dataset, num_rollouts=num_rollouts, initial_action=2,
                                 policy=policy_net, features=features, device=torch.device('cpu'),
                                 downsample_image_size=downsample_image_size))
    # Start new Threads
    for thread in threads:
        thread.start()

    print('Waiting for worker threads')

    # Wait for all threads to complete
    for t in threads:
        t.join()

    print('All threads exited')

    if epoch % 20 == 0 and save:
        torch.save(policy_net.state_dict(), 'vanilla.wgt')
    rollout_dataset.post_process()
    return rollout_dataset

@timeit
def train(policy_net, rollout, device):
    policy_net = policy_net.train().to(device)
    rollout_loader = DataLoader(rollout, batch_size=len(rollout))
    for i, (observation, reward, action, value, done) in enumerate(rollout_loader):
        action = action.to(device).float()
        value = value.to(device).float()
        observation = observation.to(device)
        action[action == 2] = 1.0
        action[action == 3] = 0.0
        optim.zero_grad()
        action_prob = policy_net(observation.squeeze().view(-1, features)).squeeze()
        prob_action_taken = action * torch.log(action_prob + 1e-12) + (1 - action) * (torch.log(1 - action_prob))
        loss = - value * prob_action_taken
        loss = loss.sum()
        loss.backward()
        optim.step()


if __name__ == '__main__':

    max_rollout_len = 3000
    downsample_image_size = (100, 80)
    features = downsample_image_size[0] * downsample_image_size[1]
    tb = SummaryWriter(f'runs/rmsprop_1e3_multi_{random.randint(0,100)}')
    tb_step = 0
    num_epochs = 600
    resume = False
    save = True
    view_games = False
    num_threads = 5
    rollouts_per_thread = 2
    envs = []
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    for env in range(num_threads):
        envs.append(gym.make('Pong-v0'))

    policy_net = PolicyNet(features).to(device)

    if resume:
        policy_net.load_state_dict(torch.load('vanilla.wgt'))

    #optim = torch.optim.Adam(lr=1e-3, params=policy_net.parameters())
    optim = torch.optim.RMSprop(lr=1e-3, params=policy_net.parameters())

    for epoch in range(num_epochs):
        rollout = collect_rollouts(policy_net, envs, features, num_threads, rollouts_per_thread)
        print(f'collected {len(rollout)} frames reward {rollout.total_reward()}')
        train(policy_net, rollout, device)