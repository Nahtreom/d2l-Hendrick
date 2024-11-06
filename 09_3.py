import torch
from torch import nn
from d2l import torch as d2l

batchsize = 256
num_epoch = 10
loss = nn.CrossEntropyLoss()

train_iter, test_iter = d2l.load_data_fashion_mnist(batchsize)

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.normal(m.weight, std=0.01)

net.apply(init_weight)

trainer = torch.optim.SGD(net.parameters(), lr=0.01)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epoch, trainer)

