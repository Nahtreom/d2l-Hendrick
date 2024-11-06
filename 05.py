import torch

A = torch.arange(20).reshape(4,5).float()
print(A) 
print(A.T)

A_sum_1 = A.sum(axis=1)
print(A_sum_1)
print(A.mean(),A.numel())

B = torch.arange(5, dtype=torch.float32)
C = torch.ones(5, dtype=torch.float32) 
print(torch.dot(B,C))
print(torch.mv(A,C))

u = torch.tensor([-3.0,4.0])
print(torch.norm(u))

D = torch.ones((2,5,4),dtype=torch.float32)
print(D.shape)
D_1 = D.sum(axis=0,keepdim=True)
print(D_1.shape)

D = torch.tensor([-3.0,4.0])
print(D.norm())