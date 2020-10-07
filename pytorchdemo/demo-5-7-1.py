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
from pytorchdemo.util import load_data_fashion_mnist, sgd, FlattenLayer, semilogy, linreg, squared_loss, train_ch3, \
  evaluate_accuracy, train_ch5
from pytorchdemo.dl2_utils import corr2d

torch.set_default_tensor_type(torch.FloatTensor)

st.title('使用重复元素的网络（VGG）')

'''
VGG提出了可以通过重复使用简单的基础块来构建深度模型的思路

## VGG块

VGG块的组成规律是：连续使用数个相同的填充为1、窗口形状为3×3的卷积层后接上一个步幅为2、窗口形状为2×2的最大池化层。卷积层保持输入的高和宽不变，而池化层则对其减半。我们使用vgg_block函数来实现这个基础的VGG块，它可以指定卷积层的数量和输入输出通道数
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with st.echo():
  def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
      if i == 0:
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
      else:
        blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
      blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 这里会使宽高减半
    return nn.Sequential(*blk)

'''
VGG网络

现在我们构造一个VGG网络。它有5个卷积块，前2块使用单卷积层，而后3块使用双卷积层。第一块的输入输出通道分别是1（因为下面要使用的Fashion-MNIST数据的通道数为1）和64，之后每次对输出通道数翻倍，直到变为512。因为这个网络使用了8个卷积层和3个全连接层，所以经常被称为VGG-11
'''
with st.echo():
  conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
  fc_features = 512 * 7 * 7  # c * w * h
  fc_hidden_units = 4096  # 任意


  def vgg11(conv_arch, fc_features, fc_hidden_units=4096):
    net = nn.Sequential()
    # 卷积层部分
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
      # 每经过一个vgg_block都会使宽高减半
      net.add_module("vgg_block_" + str(i+1), vgg_block(num_convs, in_channels, out_channels))
    # 全连接层部分
    net.add_module("fc", nn.Sequential(FlattenLayer(),
                                       nn.Linear(fc_features, fc_hidden_units),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(fc_hidden_units, fc_hidden_units),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(fc_hidden_units, 10)
                                       ))
    return net

  net = vgg11(conv_arch, fc_features, fc_hidden_units)

  X = torch.rand(1, 1, 224, 224)
  for name, blk in net.named_children():
    X = blk(X)
    name, 'output shape:', X.shape

'''
# 获取数据和训练模型
因为VGG-11计算上比AlexNet更加复杂，出于测试的目的我们构造一个通道数更小，或者说更窄的网络在Fashion-MNIST数据集上进行训练。
'''
with st.echo():
  ratio = 8
  small_conv_arch = [(1, 1, 64//ratio), (1, 64//ratio, 128//ratio), (2, 128//ratio, 256//ratio),
                     (2, 256//ratio, 512//ratio), (2, 512//ratio, 512//ratio)]
  net = vgg11(small_conv_arch, fc_features // ratio, fc_hidden_units // ratio)
  net,


  lr = st.slider(label='学习率', min_value=0.001, max_value=1.0, value=0.001, step=0.001)
  num_epochs = st.slider(label='迭代周期', min_value=5, max_value=100, value=5, step=5)
  batch_size = st.slider(label='批量', min_value=1, max_value=2560, value=64, step=32)
  # 如出现“out of memory”的报错信息，可减小batch_size或resize
  train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)

  optimizer = torch.optim.Adam(net.parameters(), lr=lr)
  train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
