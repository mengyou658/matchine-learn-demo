import torch
from time import time
import streamlit as st
from matplotlib import pyplot as plt
import numpy as np
import random
import pandas as pd
import torch.utils.data as Data
from torch import nn
# import tensorwatch as tw

st.title('从头开始定义训练模型')
# w = tw.Watcher()
# twClient = tw.WatcherClient()
# twStream = twClient.create_stream(expr='lambda d: np.sum(d.weights)')
# line_plot = tw.Visualizer(twStream, vis_type='line')

# from IPython import display
# 生成数据集
nums_inputs = 2
nums_examples = 1000
true_w = [2, -3.4]
true_b = 4.2

# 生成数据集
features = torch.from_numpy(np.random.normal(0, 1, (nums_examples, nums_inputs)))
features = features.float()
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
# 加入混淆数据
labels += torch.from_numpy(np.random.normal(0, 0.01, size=labels.size())).float()
labels = labels.float()
# 注意，features的每一行是一个长度为2的向量，而labels的每一行是一个长度为1的向量（标量）
if st.checkbox("展示部分数据图"):
  plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
  st.pyplot()
  # plt.show()


# def use_svg_display():
#   # 矢量图显示
#   display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
  # use_svg_display()
  # 设置图片尺寸
  plt.rcParams['figure.figsize'] = figsize


# 随机读取固定样本数据（）
# PyTorch提供了data包来读取数据
dataSet = Data.TensorDataset(features, labels)


def data_iterator_random(batch_size, features = None, labels = None):
  # nums_examples = len(features)
  # indices = list(range(nums_examples))
  # random.shuffle(indices)
  # for i in range(0, nums_examples, batch_size):
  #   j = torch.LongTensor(indices[i: min(i + batch_size, nums_examples)])
  #   yield features.index_select(0, j), labels.index_select(0, j)
  return Data.DataLoader(dataSet, batch_size, shuffle=True)


batch_size = 10
# for X, y in data_iterator_random(batch_size, features, labels):
#   print(X,y)
#   break

# 定义模型
'''
首先，导入torch.nn模块。实际上，“nn”是neural networks（神经网络）的缩写。顾名思义，该模块定义了大量神经网络的层。之前我们已经用过了autograd，而nn就是利用autograd来定义模型。nn的核心数据结构是Module，它是一个抽象概念，既可以表示神经网络中的某个层（layer），也可以表示一个包含很多层的神经网络。在实际使用中，最常见的做法是继承nn.Module，撰写自己的网络/层。一个nn.Module实例应该包含一些层以及返回输出的前向传播（forward）方法。下面先来看看如何用nn.Module实现一个线性回归模型。
'''


# 一种方式
class LinearNet(torch.nn.Module):
  def __init__(self, n_features):
    super(LinearNet, self).__init__()
    self.linear = torch.nn.Linear(n_features, 1)

  # 定义向前传播
  def forward(self, x):
    y = self.linear(x)
    return y


# 另外一种方式
# 事实上我们还可以用nn.Sequential来更加方便地搭建网络，Sequential是一个有序的容器，网络层将按照在传入Sequential的顺序依次被添加到计算图中。
# 写法一
net = nn.Sequential(
  nn.Linear(nums_inputs, 1)
  # 此处可以传入其他层
)
# 写法二
net = nn.Sequential()
net.add_module('linear', nn.Linear(nums_inputs, 1))
# net.add_module('')...

# 写法三
from collections import OrderedDict
net = nn.Sequential(
  OrderedDict([
    ('linear', nn.Linear(nums_inputs, 1))
    # ....
  ])
)


# 初始化模型参数
# w = torch.tensor(np.random.normal(0, 0.01, (nums_inputs, 1)), dtype=torch.float32)
# b = torch.zeros(1, dtype=torch.float32)
# w.requires_grad_(requires_grad=True)
# b.requires_grad_(requires_grad=True)
# # 定义模型
# def linreg(X, w, b):
#   return torch.mm(X.float(), w.float()) + b
# # 定义损失函数
# def squared_loss(y_hat, y):
#   return (y_hat - y.view(y_hat.size())) ** 2 / 2
# # 定义优化算法
# def sgd(params, lr, batch_size):
#   for param in params:
#     param.data -= lr * param.grad / batch_size

'''
 执行训练 训练模型
 
 迭代周期个数num_epochs和学习率lr都是超参数
 
 在实践中，大多超参数都需要通过反复试错来不断调节。
 
 虽然迭代周期数设得越大模型可能越有效，但是训练时间可能过长
'''
lr = st.slider(label='学习率', min_value=0.01, max_value=1.0, value=0.03)  # 学习率
num_epochs = st.slider(label='迭代周期', min_value=20, max_value=200, value=20, step=20)

# 初始化模型参数
nn.init.normal_(net[0].weight, mean=0, std=0.01)
nn.init.constant_(net[0].bias, val=0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)
# 定义损失函数
loss = nn.MSELoss()
# 定义优化算法
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
if st.checkbox("显示优化器参数"):
  optimizer
'''
或者为不同的子网设置不同的学习率lr
optimizer = torch.optim.SGD([
  {'params': net.subnet1.parameters(),}, # lr = 0.03
  {'params': net.subnet1.parameters(), 'lr': 0.01}, 
], lr=lr)
'''
i = 0
'''
调整学习率
for param_group in optimizer.param_groups:
  param_group['lr'] *= 0.1 # 学习率为之前的0.1倍数
'''

bar = st.progress(0)
st.subheader("损失折线图")
st_chart_loss = st.line_chart()

printLogFlag = st.checkbox("打印所损失值")

# 执行训练
for epoch in range(num_epochs):
  # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。
  # X和y分别是小批量样本的特征和标签
  # for X, y in data_iterator_random(batch_size, features, labels):
  #   l = squared_loss(linreg(X, w, b).float(), y).sum() # l是有关小批量X和y的损失值
  #   l.backward() # 小批量损失对模型参数求梯度
  #   sgd([w, b], lr, batch_size) # 使用小批量随机梯度下降迭代模型参数
  #
  #   # 不要忘了梯度清零
  #   w.grad.data.zero_()
  #   b.grad.data.zero_()
  for X, y in data_iterator_random(batch_size):
    output = net(X)
    l = loss(output, y.view(-1, 1))
    optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
    l.backward()  # 小批量损失对模型参数求梯度
    optimizer.step()

  i += 1
  if i > 100:
    i = 100
  bar.progress(i)
  dense = net[0]
  dataTmp = np.array([[l.item(), true_w[0], true_w[1], dense.weight[0][0].item(), dense.weight[0][1].item(), true_b, dense.bias[0].item()]])
  if i == 1:
    print("dataTmp", dataTmp)
  st_chart_loss.add_rows(pd.DataFrame(dataTmp, columns=['loss', 'realW1', 'realW2', 'W1', 'W2', 'realB', 'B']))
  if printLogFlag:
    st.write('epoch %d, loss %f detail %s' % (epoch + 1, l.item(), str(dataTmp)))

bar.progress(100)
dense = net[0]
print(true_w, '=', dense.weight)
print(true_b, '=', dense.bias)
true_w, '=', dense.weight
true_b, '=', dense.bias

# set_figsize()
# plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
# plt.show()
