import threading
import time
import gym
import cv2
from torchvision.transforms.functional import to_tensor
import numpy as np
import cProfile


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


class ExperienceBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.experience = []

    def add_rollout(self, rollout):
        self.lock.acquire()
        print(f'copying data from Thread {threading.current_thread().threadID}')
        self.experience.append(rollout)
        self.lock.release()

    def __len__(self):
        return len(self.experience)


def dummy_pre_process(observation):
    return observation


def downsample(observation):
    greyscale = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    greyscale = cv2.resize(greyscale, (100, 80), cv2.INTER_LINEAR)
    return greyscale


class GymWorker(threading.Thread):
    def __init__(self, threadID, name, counter, env, experience_buffer, rollouts, initial_action,
                 policy, features, device, preprocessor=None):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.env = env
        self.rollouts = rollouts
        self.buffer = []
        self.experience_buffer = experience_buffer
        self.pre_process = dummy_pre_process if preprocessor is None else preprocessor
        self.default_action = initial_action
        self.policy = policy.eval()
        self.features = features
        self.device = device

    def run(self):
        print("Starting " + self.name)

        for i in range(self.rollouts):
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

            self.experience_buffer.add_rollout(self.buffer)
            self.buffer = []


def print_time(threadName, delay, counter):
    while counter:
        time.sleep(delay)
        print("%s: %s" % (threadName, time.ctime(time.time())))
        counter -= 1


if __name__ == '__main__':

    exp_buf = ExperienceBuffer()
    threads = []

    # Create new threads
    for id in range(1, 6):
        threads.append(GymWorker(id, f"Thread-{id}", id, 'Pong-v0', exp_buf, 2, 2, preprocessor=downsample))

    # Start new Threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    print(len(exp_buf))


    print("Exiting Main Thread")
