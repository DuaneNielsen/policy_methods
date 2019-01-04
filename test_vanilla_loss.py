import torch
import torch.nn as nn

policy_net = nn.Sequential(nn.Linear(4, 1), nn.Sigmoid()).double()
optim = torch.optim.RMSprop(lr=1e-3, params=policy_net.parameters())


def update(action, value, expected):
    action = action.double()
    value = value.double()
    observation = torch.Tensor(1, 4).double()

    optim.zero_grad()
    action_prob = policy_net(observation)
    taken_action_prob = action * torch.log(action_prob + 1e-12) + (1 - action) * (torch.log(1 - action_prob))
    loss = - taken_action_prob * value
    loss = loss.sum()
    loss.backward()
    optim.step()

    update_action_prob = policy_net(observation)
    diff = update_action_prob - action_prob

    print(f'value: {value.item()} action: {action.item()} old {action_prob.item()} '
          f'update {update_action_prob.item()} diff {diff.item()} expected: {expected}')


action = torch.Tensor([[1.0]])
value = torch.Tensor([[1.0]])
update(action, value, 'increase')

action = torch.Tensor([[1.0]])
value = torch.Tensor([[-1.0]])
update(action, value, 'decrease')

action = torch.Tensor([[0.0]])
value = torch.Tensor([[1.0]])
update(action, value, 'decrease')

action = torch.Tensor([[0.0]])
value = torch.Tensor([[-1.0]])
update(action, value, 'increase')
