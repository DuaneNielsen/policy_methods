import numpy as np
import torch

for i in range(10):
    action_prob = torch.Tensor([[0.8]])
    action = 2 if np.random.uniform() < action_prob else 3
    print(action)