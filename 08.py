import random
import torch
from d2l import torch as d2l

def generate_data(w, b, data_number):
    X = torch.normal(0, 1, (data_number, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1,1))

ture_w = torch.tensor([2,-3.4])
ture_b = 4.2
features, labels = generate_data(ture_w, ture_b, 1000)

def data_iter(batchsize, features, labels):
    num_examples = len(features)
    indice = list(range(num_examples))
    random.shuffle(indice)
    for i in range(0, num_examples, batchsize):
        batch_indice = torch.tensor(
            indice[i : min(i + batchsize, num_examples)])
        yield features[batch_indice], labels[batch_indice]

w = torch.normal(0, 1, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def linear_reg(X, w, b):
    return torch.matmul(X, w) + b

def square_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, batchsize, lr):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batchsize
            param.grad.zero_()

lr = 0.03
num_epochs = 5
net = linear_reg
loss = square_loss
batchsize = 10

for epoch in range(num_epochs):
    for X, y in data_iter(batchsize, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], batchsize, lr)
    with torch.no_grad():
        train_1 = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_1.mean()):f}')

print(f'w: {w}, b: {b}')