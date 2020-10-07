'''
# softmax回归
离散数据预测问题，例如分类问题
softmax 把输出单元从一个变成了多个

预测多个的时候，每一个都会有一个置信度（最符合的概率，概率最大，则可能性最大）

'''
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
import numpy as np
import streamlit as st

sys.path.append("..")  # 为了导入上层目录的d2lzh_pytorch

from pytorchdemo.util import load_data_fashion_mnist, sgd, getDevice

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)


st.title('softmax回归--图像数据')
if st.checkbox("显示数据集数据"):
  type(train_iter), print(type(train_iter))
  len(train_iter), len(train_iter), print(len(train_iter), len(train_iter))
  feature, label = train_iter.dataset[0]
  feature.shape, label, print(feature.shape, label)  # Channel x Height X Width

'''
变量feature对应高和宽均为28像素的图像。由于我们使用了transforms.ToTensor()，所以每个像素的数值为[0.0, 1.0]的32位浮点数。需要注意的是，feature的尺寸是 (C x H x W) 的，而不是 (H x W x C)。第一维是通道数，因为数据集中是灰度图像，所以通道数为1。后面两维分别是图像的高和宽。

Fashion-MNIST中一共包括了10个类别，分别为t-shirt（T恤）、trouser（裤子）、pullover（套衫）、dress（连衣裙）、coat（外套）、sandal（凉鞋）、shirt（衬衫）、sneaker（运动鞋）、bag（包）和ankle boot（短靴）。
以下函数可以将数值标签转成相应的文本标签。
'''


def get_fashion_mnist_labels(labels):
  text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
  return [text_labels[int(i)] for i in labels]


'''
跟线性回归中的例子一样，我们将使用向量表示每个样本。已知每个样本输入是高和宽均为28像素的图像。模型的输入向量的长度是 28×28=784：该向量的每个元素对应图像中每个像素。由于图像有10个类别，单层神经网络输出层的输出个数为10，因此softmax回归的权重和偏差参数分别为784×10和1×10的矩阵。
'''
with st.echo():
  num_inputs = 784
  num_outputs = 10

  W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
  b = torch.zeros(num_outputs, dtype=torch.float)

  W.requires_grad_(requires_grad=True)
  b.requires_grad_(requires_grad=True)

if st.checkbox("展示矩阵运算示例"):
  with st.echo():
    '''同一列（dim=0）或同一行（dim=1）的元素求和'''
    X = torch.tensor([[1, 2, 3], [4, 5, 6]])
    X1 = X.sum(dim=0, keepdims=True)
    X2 = X.sum(dim=1, keepdims=True)
    X1, print(X1)
    X2, print(X2)

with st.echo():
  def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdims=True)
    return X_exp / partition # 这里应用了广播机制
if st.checkbox("展示softmax示例"):

    X = torch.rand((2,5))
    X_prob = softmax(X)
    X_prob, X_prob.sum(dim=1), print(X_prob, X_prob.sum(dim=1))

'''
### 定义模型
view函数将每张原始图像改成长度为num_inputs的向量
'''
with st.echo():
  def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)

'''
### 定义损失函数
gather函数，我们得到了2个样本的标签的预测概率
'''
if st.checkbox("展示gather函数用法"):
  with st.echo():
    y_hat = torch.tensor([[0.1,0.3,0.6], [0.3,0.2,0.5]])
    "y_hat=",y_hat
    y = torch.LongTensor([0,2])
    "y=",y
    y_view = y.view(-1, 1) # view(列,行) -1表示自动设置列数
    "y_view=", y_view
    y_hat_res = y_hat.gather(1, y_view)
    "y_hat_res=",y_hat_res

with st.echo():
  def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))

'''
### 计算分类准确率
其中y_hat.argmax(dim=1)返回矩阵y_hat每行中最大元素的索引，且返回结果与变量y形状相同。相等条件判断式(y_hat.argmax(dim=1) == y)是一个类型为ByteTensor的Tensor，我们用float()将其转换为值为0（相等为假）或1（相等为真）的浮点型Tensor
'''
with st.echo():
  def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()

if st.checkbox("展示accuracy函数用法"):
  with st.echo():
    accuracy_res = accuracy(y_hat, y)
    "accuracy_res=",accuracy_res
'''
评价net在数据集data_iter上面的准确率
'''
with st.echo():
  def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
      # 如果没指定device就使用net的device
      device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
      acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().item()
      n += y.shape[0]

    return acc_sum / n

'''
# 训练模型
'''
lr = st.slider(label='学习率', min_value=0.01, max_value=1.0, value=0.03)  # 学习率
num_epochs = st.slider(label='迭代周期', min_value=5, max_value=200, value=5, step=20)
st.subheader("损失折线图")
st_chart_loss = st.line_chart()
bar = st.progress(0)

with st.echo():
  def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None, device=None):
    process=0
    if device:
      net = net.to(device)
    print("training on ", device)
    for epoch in range(num_epochs):
      train_l_sum, train_acc_sum, n, timeStart = 0.0, 0.0, 0, time.time()
      for X, y in train_iter:
        if device:
          X = X.to(device)
          y = y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y).sum()

        # 梯度清零
        if optimizer is not None:
          optimizer.zero_grad()
        elif params is not None and params[0].grad is not None:
          for param in params:
            param.grad.data.zero_()

        l.backward()
        if optimizer is None:
          sgd(params, lr, batch_size)
        else:
          optimizer.step()  # “softmax回归的简洁实现”一节将用到


        train_l_sum += l.item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]

      process += 1
      if process > 100:
        process = 100
      bar.progress(process)
      test_acc = evaluate_accuracy(test_iter, net)
      'epoch %d, loss %.4f, train acc %.3f, test acc %.3f , time %.1f sec' % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc, time.time() - timeStart)
      dataTmp = [[float('%.4f' % (train_l_sum / n)), float('%.3f' % (train_acc_sum / n)), float('%.3f' % (test_acc))]]
      st_chart_loss.add_rows(pd.DataFrame(dataTmp, columns=['loss', 'train acc', 'test acc']))

train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr, None)
