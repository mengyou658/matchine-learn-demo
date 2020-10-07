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
from pytorchdemo.util import load_data_fashion_mnist, sgd, FlattenLayer, semilogy, linreg, squared_loss, train_ch3

st.title('过拟合解决方法--丢弃法')
'''丢弃法有一些不同的变体。本节中提到的丢弃法特指倒置丢弃法（inverted dropout）'''

with st.echo():
  num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

  W1 = torch.tensor(np.random.normal(0, 0.01, size=(num_inputs, num_hiddens1)), dtype=torch.float, requires_grad=True)
  b1 = torch.zeros(num_hiddens1, requires_grad=True)
  W2 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens1, num_hiddens2)), dtype=torch.float, requires_grad=True)
  b2 = torch.zeros(num_hiddens2, requires_grad=True)
  W3 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens2, num_outputs)), dtype=torch.float, requires_grad=True)
  b3 = torch.zeros(num_outputs, requires_grad=True)

  params = [W1, b1, W2, b2, W3, b3]

  lr = st.slider(label='学习率', min_value=0.1, max_value=200.0, value=0.5)  # 学习率
  drop_prob1 = st.slider(label='丢弃--概率一层', min_value=0.0, max_value=1.0, value=0.2)  # 学习率
  drop_prob2 = st.slider(label='丢弃--概率二层', min_value=0.0, max_value=1.0, value=0.5)  # 学习率
  num_epochs = st.slider(label='迭代周期', min_value=5, max_value=100, value=5, step=5)
  batch_size = st.slider(label='批量', min_value=256, max_value=2560, value=256, step=256)
  loss = torch.nn.MSELoss()
  bar = st.progress(0)
  st.subheader("损失折线图")
  st_chart_loss = st.line_chart()
  st.subheader("准确率图")
  st_chart_tt = st.line_chart()

  def dropout(X, drop_prob):
    X = X.float()
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    # 这种情况下把全部元素都丢弃
    if keep_prob == 0:
      return torch.zeros_like(X)
    mask = (torch.randn(X.shape) < keep_prob).float()

    return mask * X / keep_prob


  def net(X, is_training=True):
    X = X.view(-1, num_inputs)
    H1 = (torch.matmul(X, W1) + b1).relu()
    if is_training:  # 只在训练模型时使用丢弃法
      H1 = dropout(H1, drop_prob1)  # 在第一层全连接后添加丢弃层
    H2 = (torch.matmul(H1, W2) + b2).relu()
    if is_training:
      H2 = dropout(H2, drop_prob2)  # 在第二层全连接后添加丢弃层
    return torch.matmul(H2, W3) + b3

  loss = torch.nn.CrossEntropyLoss()
  train_iter, test_iter = load_data_fashion_mnist(batch_size)
  train_ch3(net, train_iter, test_iter,loss, num_epochs, batch_size, params, lr
            , None,None,
            bar, st_chart_loss, st_chart_tt)

