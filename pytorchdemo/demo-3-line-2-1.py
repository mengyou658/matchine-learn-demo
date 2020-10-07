import torch
from time import time
import streamlit as st
from matplotlib import pyplot as plt
import numpy as np
import random
import pandas as pd
st.title('从头开始定义训练模型')

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
def data_iterator_random(batch_size, features, labels):
  nums_examples = len(features)
  indices = list(range(nums_examples))
  random.shuffle(indices)
  for i in range(0, nums_examples, batch_size):
    j = torch.LongTensor(indices[i: min(i + batch_size, nums_examples)])
    yield features.index_select(0, j), labels.index_select(0, j)

batch_size = 10
# for X, y in data_iterator_random(batch_size, features, labels):
#   print(X,y)
#   break


# 初始化模型参数
w = torch.tensor(np.random.normal(0, 0.01, (nums_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

# 定义模型
def linreg(X, w, b):
  return torch.mm(X.float(), w.float()) + b

# 定义损失函数
def squared_loss(y_hat, y):
  return (y_hat - y.view(y_hat.size())) ** 2 / 2

# 定义优化算法
def sgd(params, lr, batch_size):
  for param in params:
    param.data -= lr * param.grad / batch_size

'''
 执行训练 训练模型
 
 迭代周期个数num_epochs和学习率lr都是超参数
 
 在实践中，大多超参数都需要通过反复试错来不断调节。
 
 虽然迭代周期数设得越大模型可能越有效，但是训练时间可能过长
'''
lr = st.slider(label='学习率', min_value=0.01, max_value= 1.0, value= 0.03) # 学习率
num_epochs = st.slider(label='迭代周期', min_value=20, max_value=200, value=20, step=20)
net = linreg
loss = squared_loss
i = 0

bar = st.progress(0)
st.subheader("损失")
st_chart_loss = st.line_chart()

for epoch in range(num_epochs):
  # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。
  # X和y分别是小批量样本的特征和标签
  for X, y in data_iterator_random(batch_size, features, labels):
    l = squared_loss(linreg(X, w, b).float(), y).sum() # l是有关小批量X和y的损失值
    l.backward() # 小批量损失对模型参数求梯度
    sgd([w, b], lr, batch_size) # 使用小批量随机梯度下降迭代模型参数

    # 不要忘了梯度清零
    w.grad.data.zero_()
    b.grad.data.zero_()
  i+=1
  if i > 100:
    i= 100
  bar.progress(i)
  train_l = squared_loss(linreg(features, w, b), labels)
  dataTmp = np.array([[train_l.mean().item(), true_w[0], true_w[1], w[0].item(), w[1].item(), true_b, b.item()]])
  if i == 1:
    print("dataTmp", dataTmp)
  st_chart_loss.add_rows(pd.DataFrame(dataTmp, columns=['loss', 'realW1', 'realW2', 'W1', 'W2', 'realB', 'B']))
  # st.write('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

bar.progress(100)
print(true_w, '=', w)
print(true_b, '=', b)
true_w, '=', w
true_b, '=', b




# set_figsize()
# plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
# plt.show()
