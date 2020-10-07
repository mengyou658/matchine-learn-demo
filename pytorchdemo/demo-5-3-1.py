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
from pytorchdemo.dl2_utils import corr2d

torch.set_default_tensor_type(torch.FloatTensor)

st.title('多输入通道和多输出通道')
'''
前面两节里我们用到的输入和输出都是二维数组，但真实数据的维度经常更高。例如，彩色图像在高和宽2个维度外还有RGB（红、绿、蓝）3个颜色通道。假设彩色图像的高和宽分别是h和w（像素），那么它可以表示为一个3×h×w的多维数组。我们将大小为3的这一维称为通道（channel）维。本节我们将介绍含多个输入通道或多个输出通道的卷积核。
'''
'''
## 多输入通道
'''
st.image('./resources/5.3_conv_multi_in-1332327.png')
'''(1×1+2×2+4×3+5×4)+(0×0+1×1+3×2+4×3)=56。'''

with st.echo():
  def corr2d_multi_in(X, K):
    # 沿着X和K的第0维度（通道维度）分别计算再相加
    res = corr2d(X[0, :, :], K[0, :, :])
    for i in range(1, X.shape[0]):
      res += corr2d(X[i, :, :], K[i, :, :])
    return res

'''我们可以构造图5.4中的输入数组X、核数组K来验证互相关运算的输出'''
with st.echo():
  X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], ])
  K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]], ])
  K.shape,
  corr2d_multi_in(X, K),

'''
## 多输出通道
'''
with st.echo():
  def corr2d_multi_in_out(X, K):
    # 对K的第0维度遍历，每次同输入X做互相关运算。所有的结果都使用stack函数联合在一起
    return torch.stack([corr2d_multi_in(X,k) for k in K])

  K = torch.stack([K, K+1, K+2])
  K.shape,
  res = corr2d_multi_in_out(X, K)
  res

'''
## 1X1卷积层
'''
st.image('./resources/5.3_conv_1x1-94084835.png')
'''下面我们使用全连接层中的矩阵乘法来实现1×1卷积。这里需要在矩阵乘法运算前后对数据形状做一些调整'''
with st.echo():
  def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.view(c_i, h * w)
    K = K.view(c_o, c_i)
    Y = torch.mm(K, X) # 全连接层的矩阵乘法
    return Y.view(c_o, h, w)
'''经验证，做1×1卷积时，以上函数与之前实现的互相关运算函数corr2d_multi_in_out等价'''
with st.echo():
  X = torch.rand(3,3,3)
  K = torch.rand(2,3,1,1)
  Y1 = corr2d_multi_in_out_1x1(X, K)
  Y2 = corr2d_multi_in_out(X, K)
  Y1.shape, Y2.shape,
  (Y1 - Y2).norm().item() < 1e-6,
