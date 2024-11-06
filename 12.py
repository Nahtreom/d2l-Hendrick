import torch
from torch import nn
from d2l import torch as d2l

train_num, test_num, num_input, batch_size = 20, 100, 200, 5
true_W, true_b = torch.ones((num_input,1))*0.01, 0.05
train_data = d2l.synthetic_data(true_W, true_b, train_num)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_W, true_b, test_num)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

def init_param():
    W = torch.normal(0, 1, size=(num_input, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [W, b]

def penalty(w):
    return torch.sum(w.pow(2)) / 2

def train(lambd):
    W, b = init_param()
    net, loss = lambda X: d2l.linreg(X, W, b), d2l.squared_loss
    num_epochs, lr= 100, 0.03
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y) + lambd * penalty(W)
            l.sum().backward()
            d2l.sgd([W,b], lr, batch_size)
        if epoch % 5 == 0:
            print(f'epoch {epoch + 1} loss {l.sum()}')
    return [W, b]

def test(W, b, lambd):
    net, loss = lambda X: d2l.linreg(X, W, b), d2l.squared_loss
    counter, l_sum = 0, 0
    for X, y in test_iter:
        with torch.no_grad():
            counter += 1
            l = loss(net(X), y)
            l_sum += l.sum()
    print(f'test loss {l_sum / counter}')

def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_input, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss()
    num_epochs, lr = 100, 0.03
    trainer = torch.optim.SGD([{
        'params': net[0].weight,
        'weight_decay': wd}, {
        'params': net[0].bias}], lr=lr)
    for epoch in range(num_epochs):
        for X,y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            trainer.step()
            if epoch % 5 == 0:
                print(f'epoch {epoch + 1}, loss {l}')



W_final, b_final = train(lambd=3)
print('=================================')
test(W_final, b_final, 3)
print('============= wd = 0 =============')
train_concise(0)
print('============= wd = 3 =============')
train_concise(3)
print('============= wd = 5 =============')
train_concise(5)