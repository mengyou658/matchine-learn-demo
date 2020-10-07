import sys
from collections import OrderedDict

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

torch.set_default_tensor_type(torch.FloatTensor)

st.title('二维卷积层')

'''
二维互相关运算
'''
st.image('./resources/5.1_correlation.png')
st.image('./resources/5.1_correlation1.png')
# st.text('$$0\\times0+1\\times1+3\\times2+4\\times3=19,$$')
# st.text('$$1\\times0+2\\times1+4\\times2+5\\times3=25,$$')
# st.text('$$3\\times0+4\\times1+6\\times2+7\\times3=37,$$')
# st.text('$$4\\times0+5\\times1+7\\times2+8\\times3=43.$$')
with st.echo():
  def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
      for j in range(Y.shape[1]):
        Y[i, j] = (X[i:i + h, j: j + w] * K).sum()
    return Y


  X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
  K = torch.tensor([[0, 1], [2, 3]])
  res = corr2d(X, K)
  res,

'''二维卷积层'''
with st.echo():
  class Conv2d(nn.Module):
    def __init__(self, kernel_size, ):
      super(Conv2d, self).__init__()
      self.weight = nn.Parameter(torch.randn(kernel_size))
      self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
      return corr2d(x, self.weight) + self.bias

'''
### 图像中物体边缘检测
下面我们来看一个卷积层的简单应用：检测图像中物体的边缘，即找到像素变化的位置。首先我们构造一张6×8的图像（即高和宽分别为6像素和8像素的图像）。它中间4列为黑（0），其余为白（1）。
'''
with st.echo():
  X = torch.ones(6, 8)
  X[:, 2:6] = 0
  X
'''
然后我们构造一个高和宽分别为1和2的卷积核K。当它与输入做互相关运算时，如果横向相邻元素相同，输出为0；否则输出为非0。
'''
with st.echo():
  K = torch.tensor([[1., -1.]])
'''
下面将输入X和我们设计的卷积核K做互相关运算。可以看出，我们将从白到黑的边缘和从黑到白的边缘分别检测成了1和-1。其余部分的输出全是0。
'''
with st.echo():
  Y = corr2d(X, K)
  Y,

'''
### 通过数据学习核数组
最后我们来看一个例子，它使用物体边缘检测中的输入数据X和输出数据Y来学习我们构造的核数组K。我们首先构造一个卷积层，其卷积核将被初始化成随机数组。接下来在每一次迭代中，我们使用平方误差来比较Y和卷积层的输出，然后计算梯度来更新权重。
'''
with st.echo():
  conv2d = Conv2d(kernel_size=(1, 2))

  step = st.slider(label='学习率', min_value=20, max_value=100, value=20)  # 学习率
  lr = st.slider(label='学习率', min_value=0.01, max_value=1.0, value=0.01)  # 学习率

  for i in range(step):
    y_hat = conv2d(X)
    l = ((y_hat -Y)**2).sum()
    l.backward()

    # 梯度下降
    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad

    # 梯度清零
    conv2d.weight.grad.fill_(0)
    conv2d.bias.grad.fill_(0)

    if (i + 1) % 5 == 0:
      'Step %d, loss %.3f' % (i + 1, l.item())
    

