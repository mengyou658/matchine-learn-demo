import sys

import torch
import time
import streamlit as st
from matplotlib import pyplot as plt
import numpy as np
import random
import pandas as pd
import torch.utils.data as Data
from torch import nn

# import tensorwatch as tw
from torch.nn import init

sys.path.append("..")
from pytorchdemo.util import load_data_fashion_mnist, sgd, FlattenLayer

st.title('多层感知机的简洁实现')

with st.echo():
  batch_size = 256
  train_iter, test_iter = load_data_fashion_mnist(batch_size)

  num_inputs, num_outputs, num_hiddens = 784, 10, 256

  # W1 = torch.tensor(np.random.normal(0,0.01,(num_inputs, num_hiddens)), dtype=torch.float)
  # b1 = torch.zeros(num_hiddens, dtype=torch.float)
  # W2 = torch.tensor(np.random.normal(0,0.01,(num_hiddens, num_outputs)), dtype=torch.float)
  # b2 = torch.zeros(num_outputs, dtype=torch.float)
  #
  # params = [W1, b1, W2, b2]
  # for param in params:
  #   param.requires_grad_(requires_grad=True)

'''
#### 定义激活函数
'''
with st.echo():
  def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))

'''
#### 定义模型
使用softmax回归一样
'''
with st.echo():
  # def net(X):
  #   X = X.view((-1, num_inputs))
  #   H = relu(torch.matmul(X, W1) + b1)
  #   return torch.matmul(H, W2) + b2
  net = nn.Sequential(FlattenLayer(), nn.Linear(num_inputs, num_hiddens), nn.ReLU(), nn.Linear(num_hiddens, num_outputs))
  for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)

with st.echo():
  def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
      # 如果没指定device就使用net的device
      device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
      acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().item()
      n += y.shape[0]

    return acc_sum / n

'''
#### 定义损失函数
'''
with st.echo():
  loss = torch.nn.CrossEntropyLoss()


'''
#### 开始训练
'''
with st.echo():
  lr = st.slider(label='学习率', min_value=0.01, max_value=1.0, value=0.03)  # 学习率
  optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
  num_epochs = st.slider(label='迭代周期', min_value=5, max_value=200, value=5, step=5)
  st.subheader("损失折线图")
  st_chart_loss = st.line_chart()
  st_chart_s = st.line_chart()
  bar = st.progress(0)


  def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None,
                device=None):
    process = 0
    if device:
      net = net.to(device)
    print("training on ", device)
    for epoch in range(num_epochs):
      train_l_sum, train_acc_sum, n, timeStart = 0.0, 0.0, 0, time.time()
      for X, y in train_iter:
        if device:
          X = X.to(device)
          y = y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y).sum()

        # 梯度清零
        if optimizer is not None:
          optimizer.zero_grad()
        elif params is not None and params[0].grad is not None:
          for param in params:
            param.grad.data.zero_()

        l.backward()
        if optimizer is None:
          sgd(params, lr, batch_size)
        else:
          optimizer.step()  # “softmax回归的简洁实现”一节将用到

        train_l_sum += l.item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]

      process += 1
      if process > 100:
        process = 100
      bar.progress(process)
      test_acc = evaluate_accuracy(test_iter, net)
      'epoch %d, loss %.4f, train acc %.3f, test acc %.3f , time %.1f sec' % (
      epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc, time.time() - timeStart)
      dataTmp = [[float('%.4f' % (train_l_sum / n)), ]]
      dataTmp1 = [[float('%.3f' % (train_acc_sum / n)), float('%.3f' % (test_acc))]]
      st_chart_loss.add_rows(pd.DataFrame(dataTmp, columns=['loss',]))
      st_chart_s.add_rows(pd.DataFrame(dataTmp1, columns=['train acc', 'test acc']))

train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
