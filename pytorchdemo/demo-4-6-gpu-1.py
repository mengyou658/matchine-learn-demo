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

st.title('GPU计算')

'''
计算设备
'''
with st.echo():
  torch.cuda.is_available(), # 输出 True
  torch.cuda.device_count(),
  torch.cuda.current_device(),

  torch.cuda.get_device_name(0),


'''Tensor 的GPU计算'''

with st.echo():
  x = torch.tensor([1,2,3])
  x, x.device,

  x = x.cuda(0)
  x, x.device,

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  x = torch.tensor([1, 2, 3], device=device)
  # or
  x = torch.tensor([1, 2, 3]).to(device)
  x

  # # 需要注意的是，存储在不同位置中的数据是不可以直接进行计算的。即存放在CPU上的数据不可以直接与存放在GPU上的数据进行运算，位于不同GPU上的数据也是不能直接进行计算的。
  # y = torch.tensor([1, 2, 3])
  # z  = y + x # 这里会报错

'''模型的GPU计算'''
with st.echo():
  net = nn.Linear(3,1)
  list(net.parameters())[0].device,

  net.cuda()
  list(net.parameters())[0].device,

  x = torch.rand(2,3).cuda()
  net(x),
