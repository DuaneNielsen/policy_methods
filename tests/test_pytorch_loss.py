import torch
import torch.nn.functional as NN

import torch
import torch.nn as nn

policy_net = nn.Sequential(nn.Linear(4, 2), nn.LogSoftmax(dim=1)).double()
optim = torch.optim.RMSprop(lr=1e-3, params=policy_net.parameters())


def update(action, value, expected):
    value = value.double()
    observation = torch.Tensor(1, 4).double()

    optim.zero_grad()
    action_logprob = policy_net(observation)
    loss = NN.nll_loss(action_logprob, action)
    loss *= value.squeeze()
    loss = loss.sum()
    loss.backward()
    optim.step()

    update_action_logprob = policy_net(observation)

    action_prob = torch.exp(action_logprob)
    update_action_prob = torch.exp(update_action_logprob)

    diff = update_action_prob - action_prob

    print(f'value: {value.item()} action: {action.item()} old {action_prob[0, action].item()} '
          f'update {update_action_prob[0, action].item()} diff {diff[0, action].item()} expected: {expected}')


action = torch.Tensor([1]).long()
value = torch.Tensor([[1.0]])
update(action, value, 'increase')

action = torch.Tensor([1]).long()
value = torch.Tensor([[-1.0]])
update(action, value, 'decrease')

action = torch.Tensor([0]).long()
value = torch.Tensor([[1.0]])
update(action, value, 'increase')

action = torch.Tensor([0]).long()
value = torch.Tensor([[-1.0]])
update(action, value, 'decrease')
