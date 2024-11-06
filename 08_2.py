import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

def generate_data(w, b, data_number):
    X = torch.normal(0, 1, (data_number, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1,1))

def load_data(dataarray, batchsize, is_train=True):
    dataset = data.TensorDataset(*dataarray)
    return data.DataLoader(dataset, batchsize, shuffle=is_train)


true_w = torch.tensor([2, -3.4])
true_b = 4.2
batchsize = 10
num_epochs = 3
features, labels = generate_data(true_w, true_b, 1000)

net = nn.Sequential(nn.Linear(2, 1))

net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr = 0.03)

data_iter = load_data((features, labels), batchsize)

for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f"epoch {epoch + 1}, loss {l}")
print(net.parameters)