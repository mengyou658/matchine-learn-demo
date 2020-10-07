from __future__ import print_function
import torch
x = torch.ones(2, 2, requires_grad=True)
print(x)
y = x + 2
print(y)
print(y.grad_fn)

z = y * y * 3
# 将输入所有维度为1的去除
out = z.squeeze()
# 求幂次运算 次方运算
out = z.pow(2)
print(z, out)
# 求平均
out = z.mean()
print(z, out)

print(x.is_leaf, y.is_leaf)


a = torch.randn(2,2)
a = ((a * 3)/(a-1))
print(a.requires_grad)

a.requires_grad_(True)

b = (a * a).sum()

print(a.grad_fn)
print(b.grad_fn)

# 梯度
out.backward() # 等价于 out.backward(torch.tensor(1.))
print(x.grad)
print(y.grad)
print(z.grad)

# grad在反向传播过程中是累加的(accumulated)，这意味着每一次运行反向传播，梯度都会累加之前的梯度，所以一般在反向传播之前需把梯度清零
# 再来反向传播一次，注意grad是累加的
out2 = x.sum()
out2.backward()
print(x.grad)

out3 = x.sum()
x.grad.data.zero_()
out3.backward()
print(x.grad)

# 不允许张量对张量求导，只允许标量对张量求导，求导结果是和自变量同形的张量
