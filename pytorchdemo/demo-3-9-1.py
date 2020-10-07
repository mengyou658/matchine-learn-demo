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
from pytorchdemo.util import load_data_fashion_mnist, sgd, FlattenLayer, semilogy

st.title('模型选择、欠拟合和过拟合')
'''
一类是模型无法得到较低的训练误差，我们将这一现象称作欠拟合（underfitting）；另一类是模型的训练误差远小于它在测试数据集上的误差，我们称该现象为过拟合（overfitting）
'''
st.subheader('模型复杂度对欠拟合和过拟合的影响')
st.image("./resources/3.11_capacity_vs_error-34890234.png")
st.subheader('多项式函数拟合实验')

st.text('$$y = 1.2x - 3.4x^2 + 5.6x^3 + 5 + \epsilon,$$')

# st.markdown('<script async="async" type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>', unsafe_allow_html=True)

with st.echo():
  n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
  features = torch.randn(n_train + n_test, 1)
  poly_features = torch.cat((features, torch.pow(features, 2), torch.pow(features, 3)), 1)
  labels = (
      true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1] + true_w[2] * poly_features[:, 2] + true_b)
  labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

if st.checkbox('展示数据集样本'):
  features[:2]
  poly_features[:2]
  labels[:2]



'''
#### 定义、训练和测试模型
'''
with st.echo():
  lr = st.slider(label='学习率', min_value=0.01, max_value=1.0, value=0.03)  # 学习率
  num_epochs = st.slider(label='迭代周期', min_value=5, max_value=200, value=5, step=5)
  loss = torch.nn.MSELoss()
  st.subheader("损失折线图")
  bar = st.progress(0)

  def fit_and_plot(train_features, test_features, train_labels, test_labels):
    process = 0
    net = torch.nn.Linear(train_features.shape[-1], 1)
    batch_size = min(10, train_labels.shape[0])
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    train_ls, test_ls = [], []

    for _ in range(num_epochs):
      for X, y in train_iter:
        l = loss(net(X), y.view(-1,1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

      train_labels = train_labels.view(-1,1)
      test_labels = test_labels.view(-1,1)
      train_ls.append(loss(net(train_features), train_labels).item())
      test_ls.append(loss(net(test_features), test_labels).item())

      process += 1
      if process > 100:
        process = 100
      bar.progress(process)

    'final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1]
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    'weight:', net.weight.data,'\nbias:', net.bias.data

  '''
  ### 三阶多项式函数拟合（正常）
  '''
  fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :]
               , labels[:n_train], labels[n_train:])

  '''
  ### 线性函数拟合（欠拟合）
  '''
  fit_and_plot(features[:n_train, :], features[n_train:, :], labels[:n_train],
               labels[n_train:])

  '''
  ### 线性函数拟合（过拟合）
  1. 增大训练数据集（代价高昂）
  1. 过拟合问题的常用方法：权重衰减（weight decay）
  '''
  fit_and_plot(poly_features[0:2, :], poly_features[n_train:, :], labels[0:2],
               labels[n_train:])
