from __future__ import print_function
import torch


# 不允许张量对张量求导，只允许标量对张量求导，求导结果是和自变量同形的张量

x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
y = 2 * x
z = y.view(2, 2)
print(z)

# 现在 y 不是一个标量，所以在调用backward时需要传入一个和y同形的权重向量进行加权求和得到一个标量。
v = torch.tensor([[1.0, 0.1], [0.01, 0.001]], dtype=torch.float)
z.backward(v)
print(x.grad)
