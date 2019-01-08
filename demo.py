import torch
import argparse
import gym
from viewer import UniImageViewer
import cv2
from vanilla_pong import PolicyNet
from vanilla import MultiPolicyNet
from torchvision.transforms.functional import to_tensor
import numpy as np
import time

def downsample(observation):
    greyscale = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    greyscale = cv2.resize(greyscale, downsample_image_size, cv2.INTER_LINEAR)
    return greyscale


if __name__ == '__main__':

    parser = argparse.ArgumentParser('learn to play pong')
    parser.add_argument('--reload', default='rmsprop_8vanilla.wgt')
    parser.add_argument('--speed', type=float, default=0.02)
    parser.add_argument('--multi', dest='multi', action='store_true')
    parser.set_defaults(multi=False)
    args = parser.parse_args()

    downsample_image_size = (100, 80)
    features = downsample_image_size[0] * downsample_image_size[1]
    default_action = 2
    num_rollouts = 10

    if args.multi:
        policy = MultiPolicyNet(features, [2, 3])
    else:
        policy = PolicyNet(features)

    policy.load_state_dict(torch.load(args.reload))
    env = gym.make('Pong-v0')
    v = UniImageViewer('pong', (200, 160))

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
            action = 2 if action_prob > 0.5 else 3
            #action = 2 if np.random.uniform() < action_prob.item() else 3
            observation_t1, reward, done, info = env.step(action)

            env.render(mode='human')

            # compute the observation that resulted from our action
            observation_t1 = downsample(observation_t1)
            observation = observation_t1 - observation_t0
            observation_t0 = observation_t1

            time.sleep(args.speed)

