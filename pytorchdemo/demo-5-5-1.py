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
  evaluate_accuracy
from pytorchdemo.dl2_utils import corr2d

torch.set_default_tensor_type(torch.FloatTensor)

st.title('卷积神经网络（LeNet）')

'''
* 问题
  
  在3.9节（多层感知机的从零开始实现）里我们构造了一个含单隐藏层的多层感知机模型来对Fashion-MNIST数据集中的图像进行分类。每张图像高和宽均是28像素。我们将图像中的像素逐行展开，得到长度为784的向量，并输入进全连接层中。然而，这种分类方法有一定的局限性。
    1. 图像在同一列邻近的像素在这个向量中可能相距较远。它们构成的模式可能难以被模型识别。
    1. 对于大尺寸的输入图像，使用全连接层容易造成模型过大。假设输入是高和宽均为1000像素的彩色照片（含3个通道）。即使全连接层输出个数仍是256，该层权重参数的形状是3,000,000×256：它占用了大约3 GB的内存或显存。这带来过复杂的模型和过高的存储开销。
    
* 卷积层尝试解决这两个问题

  一方面，卷积层保留输入形状，使图像的像素在高和宽两个方向上的相关性均可能被有效识别；
  
  另一方面，卷积层通过滑动窗口将同一卷积核与不同位置的输入重复计算，从而避免参数尺寸过大。
  
卷积神经网络就是含卷积层的网络。

本节里我们将介绍一个早期用来识别手写数字图像的卷积神经网络：LeNet [1]。这个名字来源于LeNet论文的第一作者Yann LeCun。LeNet展示了通过梯度下降训练卷积神经网络可以达到手写数字识别在当时最先进的结果。这个奠基性的工作第一次将卷积神经网络推上舞台，为世人所知。LeNet的网络结构如下图所示。
'''
st.image('./resources/5.5_lenet.png', width=700)

st.subheader('LeNet模型')
'''
LeNet分为卷积层块和全连接层块两个部分
  
    *. 卷积层块里的基本单位是卷积层后接最大池化层：
        
        1. 卷积层用来识别图像里的空间模式，如线条和物体局部，
        1. 之后的最大池化层则用来降低卷积层对位置的敏感性。
    卷积层块由两个这样的基本单位重复堆叠构成。在卷积层块中，每个卷积层都使用5×5的窗口，
    并在输出上使用sigmoid激活函数。第一个卷积层输出通道数为6，第二个卷积层输出通道数则增加到16。
    这是因为第二个卷积层比第一个卷积层的输入的高和宽要小，所以增加输出通道使两个卷积层的参数
    尺寸类似。卷积层块的两个最大池化层的窗口形状均为2×2，且步幅为2。由于池化窗口与步幅形状
    相同，池化窗口在输入上每次滑动所覆盖的区域互不重叠。
    卷积层块的输出形状为(批量大小, 通道, 高, 宽)。当卷积层块的输出传入全连接层块时，
    全连接层块会将小批量中每个样本变平（flatten）。也就是说，全连接层的输入形状将变
    成二维，其中第一维是小批量中的样本，第二维是每个样本变平后的向量表示，且向量长
    度为通道、高和宽的乘积。全连接层块含3个全连接层。它们的输出个数分别是120、84和10
    ，其中10为输出的类别个数。
'''

with st.echo():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # device = torch.device('cpu')
  class LeNet(nn.Module):
    def __init__(self):
      super(LeNet, self).__init__()
      self.conv = nn.Sequential(
        nn.Conv2d(1,6,5),
        nn.Sigmoid(),
        nn.MaxPool2d(2,2),
        nn.Conv2d(6, 16, 5),
        nn.Sigmoid(),
        nn.MaxPool2d(2,2)
      )
      self.fc = nn.Sequential(
        nn.Linear(16*4*4, 120),
        nn.Sigmoid(),
        nn.Linear(120, 84),
        nn.Sigmoid(),
        nn.Linear(84, 10)
      )

    def forward(self, img):
      feature = self.conv(img)
      output = self.fc(feature.view(img.shape[0], -1))
      return output

  net = LeNet()
  net,

'''测试'''
with st.echo():
  batch_size = st.slider(label='批量', min_value=256, max_value=2560, value=256, step=256)
  lr = st.slider(label='学习率', min_value=0.001, max_value=1.0, value=0.001, step=0.001)
  num_epochs = st.slider(label='迭代周期', min_value=5, max_value=100, value=5, step=5)
  train_iter, test_iter = load_data_fashion_mnist(batch_size)
  # 因为卷积神经网络计算比多层感知机要复杂，建议使用GPU来加速计算。因此，我们对3.6节（softmax回归的从零开始实现）中描述的evaluate_accuracy函数略作修改，使其支持GPU计算。

  def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    device,
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    for epoch in range(num_epochs):
      train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
      for X, y in train_iter:
        X = X.to(device)
        y = y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_l_sum += l.cpu().item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
        n += y.shape[0]
        batch_count += 1
      test_acc = evaluate_accuracy(test_iter, net)
      'epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec' % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start)

  optimizer = torch.optim.Adam(net.parameters(), lr)
  train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

