import torch
from torch import nn
from d2l import torch as d2l

batchsize = 256

train_iter, test_iter = d2l.load_data_fashion_mnist(batchsize)

num_inputs, num_outputs, num_hidden = 784, 10, 256

loss = nn.CrossEntropyLoss

num_epoch, lr = 10, 0.1

def ReLU(X):
    a = torch.zeros_like(X)
    return torch.max(a, X)

def net(X):
    X = X.reshape((-1, num_inputs))
    H = ReLU(X @ W_1 + b_1)
    return ReLU(H @ W_2 + b_2)


W_1 = nn.Parameter(torch.randn(num_inputs, num_hidden, requires_grad=True))
b_1 = nn.Parameter(torch.zeros(num_inputs, num_hidden, requires_grad=True))
W_2 = nn.Parameter(torch.randn(num_hidden, num_outputs, requires_grad=True))
b_2 = nn.Parameter(torch.zeros(num_hidden, num_outputs, requires_grad=True))


params = [W_1, b_1, W_2, b_2]

updater = torch.optim.SGD(params, lr)

