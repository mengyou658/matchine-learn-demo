import torch
from time import time

from matplotlib import pyplot as plt
import numpy as np
import random
# from IPython import display
# 生成数据集
nums_inputs = 2
nums_examples = 1000
true_w = [2, -3,4]
true_b = 4.2

features = torch.from_numpy(np.random.normal(0,1,(nums_examples, nums_inputs)))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.from_numpy(np.random.normal(0, 0.01, size=labels.size()))
# 注意，features的每一行是一个长度为2的向量，而labels的每一行是一个长度为1的向量（标量）
print(features[0], labels[0])

# def use_svg_display():
#   # 矢量图显示
#   display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5,2.5)):
  # use_svg_display()
  # 设置图片尺寸
  plt.rcParams['figure.figsize'] = figsize

set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
plt.show()
