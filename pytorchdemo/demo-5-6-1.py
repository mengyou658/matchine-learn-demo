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
import torchvision

# import tensorwatch as tw
from torch.nn import init

sys.path.append("..")
from pytorchdemo.util import load_data_fashion_mnist, sgd, FlattenLayer, semilogy, linreg, squared_loss, train_ch3, \
  evaluate_accuracy, train_ch5
from pytorchdemo.dl2_utils import corr2d

torch.set_default_tensor_type(torch.FloatTensor)

st.title('深度卷积神经网络（AlexNet）')
'''
历史原因：

在LeNet提出后的将近20年里，神经网络一度被其他机器学习方法超越，如支持向量机。虽然LeNet可以在早期的小数据集上取得好的成绩，但是在更大的真实数据集上的表现并不尽如人意。一方面，神经网络计算复杂。虽然20世纪90年代也有过一些针对神经网络的加速硬件，但并没有像之后GPU那样大量普及。因此，训练一个多通道、多层和有大量参数的卷积神经网络在当年很难完成。另一方面，当年研究者还没有大量深入研究参数初始化和非凸优化算法等诸多领域，导致复杂的神经网络的训练通常较困难。

** 计算机视觉流程中真正重要的是数据和特征。也就是说，使用较干净的数据集和较有效的特征甚至比机器学习模型的选择对图像分类结果的影响更大 ** 

## 学习特征表示

1. 不少研究者通过提出新的特征提取函数来不断改进图像分类结果
1. 然而，另外一些研究者提出异议。他们认为特征本身应该是由学习得来的

  持这一想法的研究者相信，多层神经网络可能可以学得数据的多级表征，并逐级表示越来越抽象的概念或模式。以图像分类为例，并回忆5.1节（二维卷积层）中物体边缘检测的例子。在多层神经网络中，图像的第一级的表示可以是在特定的位置和⻆度是否出现边缘；而第二级的表示说不定能够将这些边缘组合出有趣的模式，如花纹；在第三级的表示中，也许上一级的花纹能进一步汇合成对应物体特定部位的模式。这样逐级表示下去，最终，模型能够较容易根据最后一级的表示完成分类任务。需要强调的是，输入的逐级表示由多层模型中的参数决定，而这些参数都是学出来的。
  
  尽管一直有一群执着的研究者不断钻研，试图学习视觉数据的逐级表征，然而很长一段时间里这些野心都未能实现。这其中有诸多因素值得我们一一分析。

    1. 缺失要素一：数据
    1. 缺失要素二：硬件
'''

'''
## AlexNet 2012年，AlexNet横空出世
这个模型的名字来源于论文第一作者的姓名Alex Krizhevsky [1]。AlexNet使用了8层卷积神经网络，并以很大的优势赢得了ImageNet 2012图像识别挑战赛。它首次证明了学习到的特征可以超越手工设计的特征，从而一举打破计算机视觉研究的前状。
'''
st.image('./resources/5.6_alexnet.png', width=700)
'''下面我们实现稍微简化过的AlexNet'''
with st.echo():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # device = torch.device('cpu')

  print(torch.__version__)
  print(torchvision.__version__)
  print(device)


  class AlexNet(nn.Module):
    def __init__(self):
      super(AlexNet, self).__init__()
      self.conv = nn.Sequential(
        nn.Conv2d(1, 96, 11, 4), # in_channels, out_channels, kernel_size, stride, padding
        nn.ReLU(),
        nn.MaxPool2d(3, 2), # kernel_size, stride
        # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
        nn.Conv2d(96, 256, 5, 1, 2),
        nn.ReLU(),
        nn.MaxPool2d(3, 2),
        # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
        # 前两个卷积层后不使用池化层来减小输入的高和宽
        nn.Conv2d(256, 384, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(384, 384, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(384, 256, 3, 1, 1),
        nn.ReLU(),
        nn.MaxPool2d(3, 2)
      )
      # 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合
      self.fc = nn.Sequential(
        nn.Linear(256*5*5, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        # 输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
        nn.Linear(4096, 10),
      )

    def forward(self, img):
      feature = self.conv(img)
      output = self.fc(feature.view(img.shape[0], -1))
      return output

  net = AlexNet()
  net,

'''
### 读取数据
虽然论文中AlexNet使用ImageNet数据集，但因为ImageNet数据集训练时间较长，我们仍用前面的Fashion-MNIST数据集来演示AlexNet。读取数据的时候我们额外做了一步将图像高和宽扩大到AlexNet使用的图像高和宽224。这个可以通过torchvision.transforms.Resize实例来实现。也就是说，我们在ToTensor实例前使用Resize实例，然后使用Compose实例来将这两个变换串联以方便调用。
'''
with st.echo():
  batch_size = st.slider(label='批量', min_value=1, max_value=2560, value=128, step=128)
  lr = st.slider(label='学习率', min_value=0.001, max_value=1.0, value=0.001, step=0.001)
  num_epochs = st.slider(label='迭代周期', min_value=1, max_value=100, value=5, step=5)

  # 如出现“out of memory”的报错信息，可减小batch_size或resize
  train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224, num_workers=0)

  optimizer = torch.optim.Adam(net.parameters(), lr=lr)
  train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
