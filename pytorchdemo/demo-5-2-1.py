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

st.title('填充和步幅')

'''
## 填充
填充（padding）是指在输入高和宽的两侧填充元素（通常是0元素）。图5.2里我们在原输入高和宽的两侧分别添加了值为0的元素，使得输入高和宽从3变成了5，并导致输出高和宽由2增加到4。图5.2中的阴影部分为第一个输出元素及其计算所使用的输入和核数组元素：0×0+0×1+0×2+0×3=0
'''
st.image('./resources/5.2_conv_pad-57530041.png')
'''
在很多情况下，我们会设置ph=kh−1和pw=kw−1来使输入和输出具有相同的高和宽。这样会方便在构造网络时推测每个层的输出形状。假设这里kh是奇数，我们会在高的两侧分别填充ph/2行。如果kh是偶数，一种可能是在输入的顶端一侧填充⌈ph/2⌉行，而在底端一侧填充⌊ph/2⌋行。在宽的两侧填充同理。
'''
st.image('./resources/输入8x8_核心3x3输出8x8.png')
with st.echo():
  # 定义一个函数来计算卷积层。它对输入和输出做相应的升维和降维
  def comp_conv2d(conv2d, X):
    # (1, 1)代表批量大小和通道数（“多输入通道和多输出通道”一节将介绍）均为1
    # X.shape,
    X = X.view((1, 1) + X.shape)
    # X,
    # X.shape,
    Y = conv2d(X)
    # Y,
    # Y.shape,
    return Y.view(Y.shape[2:])  # 排除不关心的前两维：批量和通道


  # 注意这里是两侧分别填充1行或列，所以在两侧一共填充2行或列
  conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)

  X = torch.rand(8, 8)
  res = comp_conv2d(conv2d, X)
  res.shape,

'''
当卷积核的高和宽不同时，我们也可以通过设置高和宽上不同的填充数使输出和输入具有相同的高和宽
'''
with st.echo():
  # 使用高为5、宽为3的卷积核。在高和宽两侧的填充数分别为2和1
  conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 3), padding=(2, 1))
  comp_conv2d(conv2d, X).shape,

'''
## 步幅
'''
st.image('./resources/5.2_conv_stride-10371436.png')
'''下面我们令高和宽上的步幅均为2，从而使输入的高和宽减半。'''
with st.echo():
  conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
  comp_conv2d(conv2d, X).shape,

'''接下来是一个稍微复杂点儿的例子。'''
with st.echo():
  conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
  comp_conv2d(conv2d, X).shape,

'''
## 小结
* 填充可以增加输出的高和宽。这常用来使输出与输入具有相同的高和宽。
* 步幅可以减小输出的高和宽，例如输出的高和宽仅为输入的高和宽的1/n（n为大于1的整数）。
'''
