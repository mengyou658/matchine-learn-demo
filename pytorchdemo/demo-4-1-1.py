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

st.title('模型构造')

'''
### 模型构造 继承Module类来构造模型
'''
with st.echo():

  class MLP(nn.Module):
    def __init__(self,**kwargs):
      super(MLP,self).__init__(**kwargs)
      self.hidden = nn.Linear(784, 256) # 隐藏层
      self.act = nn.ReLU()
      self.output = nn.Linear(256, 10) # 输出层

    # 定义模型向前计算，即如果很具输入x计算返回所需要要得模型输出
    def forward(self,x):
      a = self.act(self.hidden(x))
      return self.output(a)

  X = torch.rand(2, 784)
  net = MLP()
  net
  res = net(X)
  X
  res

'''
### Module的子类
### Sequential类
'''
with st.echo():
  class MySequential(nn.Module):
    def __init__(self, *args):
      super(MySequential, self).__init__()
      if len(args) == 1 and isinstance(args[0], OrderedDict):
        for key, module in args[0].items():
          self.add_module(key, module)
      else:
        for idx, module in enumerate(args):
          self.add_module(str(idx), module)

    def forward(self, input):
      for module in self._modules.values():
        input = module(input)
      return input

  net = MySequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
  )
  net
  res = net(X)
  res

'''
### ModuleList类
'''
with st.echo():
  net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
  net.append(nn.Linear(256, 10))
  net[-1]
  net

'''
### ModuleDict类
'''
with st.echo():
  net = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU(),
  })
  net['output'] = nn.Linear(256, 10)
  net['linear']
  net.output
  net

'''
### 构造复杂的模型
'''
with st.echo():
  class FancyMLP(nn.Module):
    def __init__(self,**kwargs):
      super(FancyMLP, self).__init__(**kwargs)
      self.rand_weight = torch.rand((20,20), requires_grad=False) # 不可训练得参数（常熟参数）
      self.linear = nn.Linear(20,20)

    def forward(self,x):
      x = self.linear(x)
      # 使用创建的常数参数，以及nn.functional中的relu函数和mm函数
      x = nn.functional.relu(torch.mm(x, self.rand_weight.data) + 1)

      # 复用全连接层，等价于两个全连接层共享参数
      x = self.linear(x)

      # 控制流，这里我们需要调用item函数来返回标量进行比较
      while x.norm().item() > 1:
        x /= 2
      if x.norm().item() < 0.8:
        x *= 10

      return x.sum()

  X = torch.ones(2, 20)
  X
  net = FancyMLP()
  net
  res = net(X)
  res

  class NestMLP(nn.Module):
    def __init__(self,**kwargs):
      super(NestMLP, self).__init__(**kwargs)
      self.net = nn.Sequential(nn.Linear(40, 30), nn.ReLU())
    def forward(self, x):
      return self.net(x)
  net = nn.Sequential(NestMLP(), nn.Linear(30,20), FancyMLP())

  x = torch.rand(2, 40)
  net
  print(net)
  res = net(x)
  res
