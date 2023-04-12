import torch
import torch.nn as nn
import torch.nn.functional as F
from loss_func import *
# data = torch.ones((97,36))
# data2 = torch.LongTensor(36)
# data3 = data2*data
# print(data.shape)
# print(data2.shape)
# print(data3.shape)
# print(data)
# print(data2)
# print(data3)

# data=torch.tensor([1,2,3],dtype=torch.int64)
# data=F.one_hot(data)
# print(data)
# data=F.softmax(data.double(),dim=-1)
# print(data)

input = torch.randn(97, 36, requires_grad=True)
target = torch.randint(36, (97,), dtype=torch.int64)

# loss = loss_func(input, target)
# print(loss)

target = F.one_hot(target,num_classes=36).float()
loss = loss_func(input, target)
print(loss)
# .double().softmax(dim=-1)
# print(target)
# print(F.softmax(input,-1))
F.focal_loss
loss = F.cross_entropy(input, target)
print(loss)
loss = F.binary_cross_entropy_with_logits(input, target)
print(loss)

# input = torch.randn(3, 5, requires_grad=True)
# target = torch.randn(3, 5).softmax(dim=1)
# print(input)
# print(target)
# loss = F.cross_entropy(input, target)
# loss.backward()
# print(loss)