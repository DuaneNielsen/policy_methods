import torch
import torch.nn as nn

policy_net = nn.Sequential(nn.Linear(4, 1), nn.Sigmoid())
optim = torch.optim.RMSprop(lr=1e-3, params=policy_net.parameters())


def update(action, value, expected):
    observation = torch.Tensor(1, 4)

    optim.zero_grad()
    action_prob = policy_net(observation)
    taken_action_prob = action * action_prob + (1 - action) * (1 - action_prob)
    loss = - taken_action_prob * value
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
