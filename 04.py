import torch
import numpy
import os

a = torch.arange(12)
print(a)

X = a.reshape(3,4)
print(X)

x = torch.ones((2,3,4))
print(x)

A = a.reshape(3,4)
B = torch.zeros((3,4))

#按照不同轴进行合并
print(torch.cat((A,B),dim=0))
print(torch.cat((A,B),dim=1))

#逻辑运算和按元素求和
print(A == B)
print(A.sum())

#广播机制
C = torch.arange(3).reshape(3,1)
D = torch.arange(2).reshape(1,2)
print(C+D)

print(C.numpy())

