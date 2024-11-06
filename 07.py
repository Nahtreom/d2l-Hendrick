import torch

x = torch.arange(4.0,requires_grad=True)
print(x.grad)
y = 2 * torch.dot(x,x)
y.backward()
print(x.grad)
print(x.grad == 4*x)
x.grad.zero_()
print(x.grad)
y = x.sum()
print(y)
y.backward()
print(x.grad)
