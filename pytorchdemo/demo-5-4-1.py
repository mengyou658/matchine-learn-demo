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

st.title('池化层')

'''
回忆一下，在5.1节（二维卷积层）里介绍的图像物体边缘检测应用中，我们构造卷积核从而精确地找到了像素变化的位置。设任意二维数组X的i行j列的元素为X[i, j]。如果我们构造的卷积核输出Y[i, j]=1，那么说明输入中X[i, j]和X[i, j+1]数值不一样。这可能意味着物体边缘通过这两个元素之间。但实际图像里，我们感兴趣的物体不会总出现在固定位置：即使我们连续拍摄同一个物体也极有可能出现像素位置上的偏移。这会导致同一个边缘对应的输出可能出现在卷积输出Y中的不同位置，进而对后面的模式识别造成不便。

在本节中我们介绍池化（pooling）层，它的提出是为了缓解卷积层对位置的过度敏感性。
'''
st.subheader('二维最大池化层和平均池化层')
st.image('./resources/pooling-33823402.png')
'''
max(0,1,3,4)=4, max(1,2,4,5)=5, max(3,4,6,7)=7, max(4,5,7,8)=8. 

二维平均池化的工作原理与二维最大池化类似，但将最大运算符替换成平均运算符。池化窗口形状为p×q的池化层称为p×q池化层，其中的池化运算叫作p×q池化。

'''
with st.echo():
  def pool2d(X, pool_size, mode='max'):
    X = X.float()
    p_h, p_w = pool_size
    Y = torch.zeros(X.shape[0] - p_h + 1, X.shape[1] - p_w + 1)
    for i in range(Y.shape[0]):
      for j in range(Y.shape[1]):
        if mode == 'max':
          Y[i, j] = X[i:i + p_h, j:j + p_w].max()
        elif mode == 'avg':
          Y[i, j] = X[i:i + p_h, j:j + p_w].mean()
    return Y

'''我们可以构造图5.6中的输入数组X来验证二维最大池化层的输出'''
with st.echo():
  X = torch.tensor([[0,1,2],[3, 4, 5],[6, 7, 8]])
  res = pool2d(X, (2,2))
  res,

'''同时我们实验一下平均池化层'''
with st.echo():
  res = pool2d(X, (2,2), 'avg')
  res,


st.subheader('填充和步幅')
'''
同卷积层一样，池化层也可以在输入的高和宽两侧的填充并调整窗口的移动步幅来改变输出形状。池化层填充和步幅与卷积层填充和步幅的工作机制一样。我们将通过nn模块里的二维最大池化层MaxPool2d来演示池化层填充和步幅的工作机制。我们先构造一个形状为(1, 1, 4, 4)的输入数据，前两个维度分别是批量和通道。
'''
with st.echo():
  X = torch.arange(16, dtype=torch.float).view((1,1,4,4))
  '输出',X

'''默认情况下，MaxPool2d实例里步幅和池化窗口形状相同。下面使用形状为(3, 3)的池化窗口，默认获得形状为(3, 3)的步幅'''
with st.echo():
  pool2d = nn.MaxPool2d(3)
  res = pool2d(X)
  res,

'''我们可以手动指定步幅和填充'''
with st.echo():
  pool2d2 = nn.MaxPool2d(3, padding=1, stride=2)
  res = pool2d2(X)
  res,

'''当然，我们也可以指定非正方形的池化窗口，并分别指定高和宽上的填充和步幅'''
with st.echo():
  pool2d2 = nn.MaxPool2d((2, 4), padding=(1, 2), stride=(2, 3))
  res = pool2d2(X)
  res,

st.subheader('多通道')
'''
在处理多通道输入数据时，池化层对每个输入通道分别池化，而不是像卷积层那样将各通道的输入按通道相加。这意味着池化层的输出通道数与输入通道数相等。下面将数组X和X+1在通道维上连结来构造通道数为2的输入。
'''
with st.echo():
  X = torch.cat((X, X+1), dim=1)
  pool2d2 = nn.MaxPool2d(3, padding=1, stride=2)
  res = pool2d2(X)
  res,
