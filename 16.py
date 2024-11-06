import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
    
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            self._modules[block] = block
    
    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
#x = torch.rand(2, 20)
#print(net[2].state_dict())
#print(net[2].weight)
#print(net[2].bias)
#print(net[2].bias.grad)

def block1():
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())
    return net

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block {i}', block1())
    return net

#rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
#print(rgnet)

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight)
        nn.init.zeros_(m.bias)

net.apply(init_normal)
#print(net[2].weight)
#print(net[2].bias)

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X):
        return X - X.mean()
layer = CenteredLayer()
#print(layer(torch.FloatTensor([1,2,3,4,5])))

class MyLinear(nn.Module):
    def __init__(self, inunits, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(inunits, units))
        self.bias = nn.Parameter(torch.randn(units, 1))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
    
x = torch.arange(4)
y = torch.zeros(4)
torch.save([x, y], 'xy-file')
x1, y1 = torch.load('xy-file')
#print(x1, y1)

net1 = MLP()
X = torch.randn(size=(2, 20))
Y = net1(X)
torch.save(net1.state_dict(), 'mlp-params')
clone = MLP()
clone.load_state_dict(torch.load('mlp-params'))
print(clone.eval())
Y_clone = clone(X)
print(Y == Y_clone)