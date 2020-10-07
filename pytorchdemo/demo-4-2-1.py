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

st.title('模型参数的访问、初始化和共享')

'''
'''
with st.echo():
  net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))
  net
  X = torch.rand(2, 4)
  Y = net(X).sum()
  Y
  '''因为Parameter是Tensor，即Tensor拥有的属性它都有，比如可以根据data来访问参数数值，用grad来访问参数梯度。'''
  weight_0 = list(net[0].parameters())[0]
  weight_0.data,
  weight_0.grad,
  Y.backward()
  weight_0.grad,

'''### 访问模型参数'''
with st.echo():
  t = type(net.named_parameters())
  t

  for name, param in net.named_parameters():
    name, param.size(), type(param)

'''另外返回的param的类型为torch.nn.parameter.Parameter，其实这是Tensor的子类，
和Tensor不同的是如果一个Tensor是Parameter，那么它会自动被添加到模型的参数列表里，来看下面这个例子

'''
with st.echo():
  class MyModel(nn.Module):
    def __init__(self, **kwargs):
      super(MyModel, self).__init__(**kwargs)
      self.weight1 = nn.Parameter(torch.rand(20, 20))
      self.weight2 = torch.rand(20, 20)

    def forward(self, x):
      pass


  n = MyModel()

  '''代码中weight1在参数列表中但是weight2却没在参数列表中'''
  for name, param in n.named_parameters():
    name, param.size(), type(param)

'''
### 初始化模型参数
'''
'''初始化参数torch的实现 torch.nn.init.normal_'''
with st.echo():
  def normal_(tensor, mean=0, std=1):
    with torch.no_grad():
      return tensor.normal_(mean, std)

'''自定义实现'''
with st.echo():
  def init_weight(tensor):
    with torch.no_grad():
      tensor.uniform_(-10, 10)
      tensor *= (tensor.abs() >= 5).float()


  for name, param in net.named_parameters():
    if 'weight' in name:
      init_weight(param)
      name, param.data
'''参考2.3.2节，我们还可以通过改变这些参数的data来改写模型参数值同时不会影响梯度'''
with st.echo():
  for name, param in net.named_parameters():
    if 'bias' in name:
      param.data += 1
      name, param.data
'''
###共享模型参数
1. 4.1.3节提到了如何共享模型参数: Module类的forward函数里多次调用同一个层
1. 如果我们传入Sequential的模块是同一个Module实例的话参数也是共享的
'''
with st.echo():
  linear = nn.Linear(1, 1, bias=False)
  net = nn.Sequential(linear, linear)
  net

  for name, param in net.named_parameters():
    init.constant_(param, val=3)
    name, param.data
    id(net[0]) == id(net[1])
    id(net[0].weight == id(net[1].weight))

'''模型参数里包含了梯度，所以在反向传播计算时，这些共享的参数的梯度是累加的'''
with st.echo():
  x = torch.ones(1,1)
  y = net(x).sum()
  y
  y.backward()
  net[0].weight.grad
