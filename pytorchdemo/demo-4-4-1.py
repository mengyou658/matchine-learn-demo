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

st.title('自定义层')

'''
不含模型参数的自定义层
'''
with st.echo():
  class CenteredLayer(nn.Module):
    def __init__(self, **kwargs):
      super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
      return x - x.mean()


  layer = CenteredLayer()
  layer
  for name, param in layer.named_parameters():
    name, param.size(), type(param)
  res = layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float))
  res

  net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
  y = net(torch.rand(4, 8))
  y.mean().item(),

'''含模型参数的定义'''
with st.echo():
  class MyDense(nn.Module):
    def __init__(self):
      super(MyDense, self).__init__()
      self.params = nn.ParameterList([nn.Parameter(torch.randn(4, 4)) for i in range(3)])
      self.params.append(nn.Parameter(torch.randn(4, 1)))

    def forward(self, x):
      for i in range(len(self.params)):
        x = torch.mm(x, self.params[i])
      return x


  net = MyDense()
  net

'''而ParameterDict接收一个Parameter实例的字典作为输入然后得到一个参数字典'''
with st.echo():
  class MyDenseDict(nn.Module):
    def __init__(self, ):
      super(MyDenseDict, self).__init__()
      self.params = nn.ParameterDict({
        'linear1': nn.Parameter(torch.randn(4, 4))
        , 'linear2': nn.Parameter(torch.randn(4, 1))
      })
      self.params.update({'linear3': nn.Parameter(torch.randn(4, 2))})

    def forward(self, x, choice='linear1'):
      return torch.mm(x, self.params[choice])

  net = MyDenseDict()
  net

  x = torch.ones(1,4)
  net(x, 'linear1'),
  net(x, 'linear2'),
  net(x, 'linear3'),

