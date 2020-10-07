import sys

import torch
import time
import streamlit as st
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import random
import pandas as pd
import torch.utils.data as Data
from torch import nn

# import tensorwatch as tw
from torch.nn import init

sys.path.append("..")
from pytorchdemo.util import load_data_fashion_mnist, sgd, FlattenLayer, semilogy, linreg, squared_loss

st.title('过拟合解决方法--权重衰减--简洁实现')
# st.markdown('<script async="async" type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>', unsafe_allow_html=True)
st.text('$$y = 0.05 + \sum_{i = 1}^p 0.01x_i +  \epsilon$$')

with st.echo():
  n_train, n_test, num_inputs = 20, 100, 200
  true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05

  features = torch.randn((n_train + n_test, num_inputs))
  labels = torch.matmul(features, true_w) + true_b
  labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
  train_features, test_features = features[:n_train, :], features[n_train:, :]
  train_labels, test_labels = labels[:n_train], labels[n_train:]
if st.checkbox('展示数据集样本'):
  features[:2]
  train_features[:2]
  labels[:2]

'''
#### 定义、训练和测试模型
'''
with st.echo():
  lr = st.slider(label='学习率', min_value=0.003, max_value=1.0, value=0.003)  # 学习率
  weight_decay = st.slider(label='权重衰减-权重', min_value=0, max_value=10, value=0)  # 学习率
  num_epochs = st.slider(label='迭代周期', min_value=100, max_value=1000, value=100, step=100)
  loss = torch.nn.MSELoss()
  st.subheader("损失折线图")
  bar = st.progress(0)


  def init_params():
    w = torch.randn((num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

  '''定义L2范数惩罚项'''
  def l2_penalty(w):
    return (w ** 2).sum() / 2


  batch_size = 1
  net, loss = linreg, squared_loss
  dataset = torch.utils.data.TensorDataset(train_features, train_labels)
  train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

  def fit_and_plot(wd):
    process = 0
    train_ls, test_ls = [], []
    net = nn.Linear(num_inputs, 1)
    nn.init.normal_(net.weight, mean=0, std=1)
    nn.init.normal_(net.bias, mean=0, std=1)
    optimizer_w = torch.optim.SGD(params=[net.weight],lr=lr, weight_decay=wd)
    optimizer_b = torch.optim.SGD(params=[net.bias], lr=lr)

    for _ in range(num_epochs):
      for X, y in train_iter:
        l = loss(net(X), y).mean()
        optimizer_w.zero_grad()
        optimizer_b.zero_grad()

        l.backward()

        optimizer_w.step()
        optimizer_b.step()

      train_ls.append(loss(net(train_features), train_labels).mean().item())
      test_ls.append(loss(net(test_features), test_labels).mean().item())

      process += 1
      if process > 100:
        process = 100
      bar.progress(process)

    'final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1]
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    'L2 norm of w:', net.weight.norm().item()


  fit_and_plot(weight_decay)
