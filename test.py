import torch

n_drone = 5
print(torch.arange(n_drone))
print(torch.arange(n_drone).view(1, -1, 1).shape)
t = torch.arange(n_drone).view(1, -1, 1)

a = torch.tensor([[1,2,3]])
print(a)
print(a.unsqueeze(1) == t)
print((a.unsqueeze(1) == t).shape)