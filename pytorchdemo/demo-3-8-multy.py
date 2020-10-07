import torch
from time import time
import streamlit as st
from matplotlib import pyplot as plt
import numpy as np
import random
import pandas as pd
import torch.utils.data as Data
from torch import nn
# import tensorwatch as tw

st.title('多层感知机')

'''
# 激活函数
## ReLU函数
'''

def xyplot(x_vals, y_vals, name=None):
  fig1 = plt.figure()
  plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
  plt.xlabel('x')
  plt.ylabel(name + ' x')
  st.pyplot(fig1)

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = x.relu()
xyplot(x, y, 'relu')

'''
#### ReLU函数的导数
'''
y.sum().backward()
xyplot(x, x.grad, 'grad of relu')

'''
## sigmoid函数
'''
y = x.sigmoid()
xyplot(x, y, 'sigmoid')

'''
#### sigmoid函数的导数
'''
x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, 'grad of sigmoid')

'''
## tanh函数
'''
y = x.tanh()
xyplot(x, y, 'tanh')

'''
#### tanh函数的导数
'''
x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, 'grad of tanh')
