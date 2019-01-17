from unittest import TestCase
from ppo_clip_discrete import PolicyNet, MultiPolicyNet, nll_value_loss, binary_value_loss, train_policy
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from viewer import UniImageViewer
import random
import numpy as np


class WrappedMNIST:
    def __init__(self):
        self.mnist = MNIST('data', transform=ToTensor(), download=True)

    def __getitem__(self, item):
        return self.mnist[item][0], 1.0, self.mnist[item][1], 1.0

    def __len__(self):
        return len(self.mnist)

    def features(self):
        size = self.mnist[0][0].size()
        return size[0] * size[1] * size[2]


class Wrapped2DigitMNIST:
    def __init__(self, reward=1.0, reward_tensor=None):
        self.mnist = MNIST('data', transform=ToTensor(), download=True)
        twomnist = []
        for item in self.mnist:
            if item[1] < 2:
                twomnist.append(item)
        self.mnist = twomnist
        self.reward = reward
        self.reward_tensor = reward_tensor

    def __getitem__(self, item):
        if self.reward_tensor is not None:
            return self.mnist[item][0], self.reward_tensor[item], self.mnist[item][1], self.reward_tensor[item]
        else:
            return self.mnist[item][0], self.reward, self.mnist[item][1], self.reward

    def __len__(self):
        return len(self.mnist)

    def features(self):
        size = self.mnist[0][0].size()
        return size[0] * size[1] * size[2]


class TestLoss(TestCase):
    def run_binary_loss(self, action, value):
        observation = torch.rand(1, 200)
        policy = PolicyNet(200)
        optim = torch.optim.RMSprop(params=policy.parameters(), lr=1e-3)

        optim.zero_grad()
        action_prob = policy(observation)
        loss = binary_value_loss(action_prob, action, value)
        loss.backward()
        optim.step()

        new_action_prob = policy(observation)

        print(new_action_prob, action_prob)

        grad = new_action_prob - action_prob

        return grad

    def test_binary_loss(self):
        grad = self.run_binary_loss(torch.Tensor([1]), torch.Tensor([1.0]))
        assert grad > 0
        grad = self.run_binary_loss(torch.Tensor([1]), torch.Tensor([-1.0]))
        assert grad < 0
        grad = self.run_binary_loss(torch.Tensor([0]), torch.Tensor([1.0]))
        assert grad < 0
        grad = self.run_binary_loss(torch.Tensor([0]), torch.Tensor([-1.0]))
        assert grad > 0

    def run_nll_loss(self, action, value):
        observation = torch.rand(1, 200)
        policy = MultiPolicyNet(200, [0, 1])
        optim = torch.optim.RMSprop(params=policy.parameters(), lr=1e-3)

        optim.zero_grad()
        action_prob = policy(observation)
        loss = nll_value_loss(action_prob, action, value)
        loss.backward()
        optim.step()

        new_action_prob = policy(observation)

        new = torch.exp(new_action_prob)
        old = torch.exp(action_prob)

        print(new, old)

        grad = new - old

        return grad

    def test_nll_loss(self):

        # if action 1 is performed and rewarded, the probability should increase
        grad = self.run_nll_loss(torch.Tensor([1]).long(), torch.Tensor([1.0]))
        assert grad[0, 1] > 0
        # and action 0 should decrease in probability
        assert grad[0, 0] < 0

        # if action 1 is performed and penalized, the probability should decrease
        grad = self.run_nll_loss(torch.Tensor([1]).long(), torch.Tensor([-1.0]))
        assert grad[0, 1] < 0

        # same goes for action 0
        grad = self.run_nll_loss(torch.Tensor([0]).long(), torch.Tensor([1.0]))
        assert grad[0, 0] > 0
        grad = self.run_nll_loss(torch.Tensor([0]).long(), torch.Tensor([-1.0]))
        assert grad[0, 0] < 0

    def test_sampled_nll_loss(self):

        def compute_gradient(value):
            observation = torch.rand(1, 200)
            policy = MultiPolicyNet(200, [0, 1])

            action_prob = policy(observation)
            index, action = policy.sample(action_prob)

            optim = torch.optim.RMSprop(params=policy.parameters(), lr=1e-3)
            optim.zero_grad()
            action_prob = policy(observation)
            loss = nll_value_loss(action_prob, index, value)
            loss.backward()
            optim.step()

            new_action_prob = policy(observation)

            new = torch.exp(new_action_prob[0, index]).item()
            old = torch.exp(action_prob[0, index]).item()

            return new - old

        for _ in range(10):
            assert compute_gradient(1.0) > 0

        for _ in range(10):
            assert compute_gradient(-1.0) < 0

    def test_nll_with_MNIST(self):
        mnist = WrappedMNIST()
        net = MultiPolicyNet(mnist.features(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        optim = torch.optim.Adam(net.parameters())

        for epoch in range(60):
            train_policy(net, mnist, optim)

        v = UniImageViewer('mnist digits')

        for sample in random.sample(range(len(mnist)), 10):
            prob = net(mnist[sample][0].unsqueeze(0).view(-1, net.features))
            print(torch.exp(prob))
            print(torch.argmax(torch.exp(prob), dim=1))
            v.render(mnist[sample][0], block=True)

    def run2digit(self, reward, visualize=False):
        mnist = Wrapped2DigitMNIST(reward)
        net = MultiPolicyNet(mnist.features(), [0, 1])
        optim = torch.optim.Adam(net.parameters())
        for epoch in range(60):
            train_policy(net, mnist, optim)
        v = UniImageViewer('mnist digits')
        correct = 0
        for sample in random.sample(range(len(mnist)), 10):
            prob = net(mnist[sample][0].unsqueeze(0).view(-1, net.features))
            if visualize:
                print(torch.exp(prob))
                print(torch.argmax(torch.exp(prob), dim=1))
                v.render(mnist[sample][0])
            if torch.argmax(torch.exp(prob)).item() == mnist[sample][2].item():
                correct += 1
        return correct

    def test_nll_with_MNIST_reward_one(self):
        correct = self.run2digit(1.0)
        print(correct)
        assert correct >= 8

    def test_nll_with_MNIST_reward_minus_one(self):
        correct = self.run2digit(-1.0)
        print(correct)
        assert correct <= 1

    def test_nll_with_MNIST_reward_five(self):
        correct = self.run2digit(5.0)
        print(correct)
        assert correct >= 8

    def test_nll_with_MNIST_reward_minus_five(self):
        correct = self.run2digit(-5.0)
        print(correct)
        assert correct <= 1

    def test_with_samples(self):

        def run_test():
            mnist_2 = Wrapped2DigitMNIST()
            loader = DataLoader(mnist_2, batch_size=len(mnist_2))
            policy = MultiPolicyNet(mnist_2.features(), [0, 1])
            reward = torch.zeros(len(mnist_2), dtype=torch.float64)

            for image, _, target, value in loader:
                logprobs = policy(image.view(-1, policy.features))
                index, action = policy.sample(logprobs)

                # rand = np.random.random(logprobs.size(0))
                # probs = torch.exp(logprobs)
                # index = (probs[:, 0].detach().numpy() > rand) * 1
                # index = torch.from_numpy(index).long()

                reward[target == index] = 1.0
                reward[target != index] = -1.0

            mnist = Wrapped2DigitMNIST(reward_tensor=reward)
            net = MultiPolicyNet(mnist.features(), [0, 1])
            optim = torch.optim.SGD(net.parameters(), lr=1e-9)

            train_policy(net, mnist, optim)

            v = UniImageViewer('mnist digits')
            correct = 0

            visualize = False
            for sample in random.sample(range(len(mnist)), 100):
                prob = net(mnist[sample][0].unsqueeze(0).view(-1, net.features))
                if visualize:
                    print(torch.exp(prob))
                    print(torch.argmax(torch.exp(prob), dim=1))
                    v.render(mnist[sample][0])
                if torch.argmax(torch.exp(prob)).item() == mnist[sample][2].item():
                    correct += 1
            return correct

        for epoch in range(60):
            correct = run_test()
            print(correct)
        assert correct >= 8
