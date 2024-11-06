import torch
from torch import nn
from d2l import torch as d2l

def cord2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return cord2d(x, self.weight)
    
X = torch.ones((4, 8))
X[:, 2:6] = 0
K = torch.tensor([[1, -1]])
Y = cord2d(X, K)

conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
X = X.reshape((1, 1, 4, 8))
Y = Y.reshape((1, 1, 4, 7))

lr = 3e-2
num_epoch = 15

for epoch in range(num_epoch):
    y_hat = conv2d(X)
    l = (y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    print(f'epoch {epoch + 1}, loss {l.sum()}')

print('=====================')
print(f'final weight {conv2d.weight.data[:]}')