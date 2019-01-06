from unittest import TestCase
from vanilla import PolicyNet, MultiPolicyNet, nll_value_loss,  binary_value_loss
import torch


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
        policy = MultiPolicyNet(200)
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
