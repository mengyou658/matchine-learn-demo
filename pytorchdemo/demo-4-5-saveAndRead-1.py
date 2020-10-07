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

st.title('读取和存储')

'''
读写Tensor
'''
with st.echo():
  x = torch.ones(3)
  torch.save(x, 'x.pt')

  x2 = torch.load('x.pt')
  x2

  y = torch.zeros(4)
  torch.save([x, y], 'xy.pt')

  xy_list = torch.load('xy.pt')
  xy_list[0], xy_list[1]

'''读写模型'''
'''
###### state_dict
在PyTorch中，Module的可学习参数(即权重和偏差)，模块模型包含在参数中(通过model.parameters()访问)。state_dict是一个从参数名称隐射到参数Tesnor的字典对象
'''
with st.echo():
  class MLP(nn.Module):
    def __init__(self,):
      super(MLP, self).__init__()
      self.hidden = nn.Linear(3,2)
      self.act = nn.ReLU()
      self.output = nn.Linear(2,1)

    def forward(self,x):
      a = self.act(self.hidden(x))
      return self.output(a)

  net = MLP()
  net.state_dict(),

  optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
  optimizer.state_dict(),


'''
### 保存和加载模型
1. 仅仅保存和加载模型参数（state_dict）
1. 保存和加载整个模型
'''
'''##### 保存和加载模型参数（state_dict）'''
'''1. 保存：'''
with st.echo():
  torch.save(net.state_dict(), 'test-model.pt')

'''1. 加载：'''
with st.echo():
  net.load_state_dict(torch.load('test-model.pt'))

'''##### 保存和加载整个模型'''
'''1. 保存'''
with st.echo():
  torch.save(net, 'test-model-all.pt')

'''1. 加载'''
with st.echo():
  net = torch.load('test-model-all.pt')

'''测试'''
with st.echo():
  X = torch.randn(2,3)
  Y = net(X)

  PATH = './net.pt'
  torch.save(net.state_dict(), PATH)

  net2 = MLP()
  net2.load_state_dict(torch.load(PATH))
  Y2 = net2(X)
  (Y2 == Y).float(),

