# set_figsize()
# plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
# plt.show()
import torch
from time import time
import streamlit as st
from matplotlib import pyplot as plt
import numpy as np
import random

# from IPython import display
nums_inputs = 2
nums_examples = 1000
true_w = [2, -3.4]
true_b = 4.2

features = torch.from_numpy(np.random.normal(0, 1, (nums_examples, nums_inputs)))
features = features.float()
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.from_numpy(np.random.normal(0, 0.01, size=labels.size())).float()
labels = labels.float()
if st.checkbox("show data demo"):
  st.write(features[0])
  st.write(labels[0])


# def use_svg_display():
#   display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
  # use_svg_display()
  plt.rcParams['figure.figsize'] = figsize

def data_iterator_random(batch_size, features, labels):
  nums_examples = len(features)
  indices = list(range(nums_examples))
  random.shuffle(indices)
  for i in range(0, nums_examples, batch_size):
    j = torch.LongTensor(indices[i: min(i + batch_size, nums_examples)])
    yield features.index_select(0, j), labels.index_select(0, j)

batch_size = 10
for X, y in data_iterator_random(batch_size, features, labels):
  print(X,y)
  break


w = torch.tensor(np.random.normal(0, 0.01, (nums_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

def linreg(X, w, b):
  return torch.mm(X.float(), w.float()) + b

def squared_loss(y_hat, y):
  return (y_hat - y.view(y_hat.size())) ** 2 / 2

def sgd(params, lr, batch_size):
  for param in params:
    param.data -= lr * param.grad / batch_size


st.title('define learning')
lr = st.slider(label='learning rate', min_value=0.01, max_value= 1.0, value= 0.03)
num_epochs = st.slider(label='batch_size', min_value=20, max_value=200, value=20, step=20)
net = linreg
loss = squared_loss
i = 0
bar = st.progress(0)
st_chart = st.line_chart(np.array([0, 0.00]))
for epoch in range(num_epochs):
  for X, y in data_iterator_random(batch_size, features, labels):
    l = squared_loss(linreg(X, w, b).float(), y).sum()
    l.backward()
    sgd([w, b], lr, batch_size)

    w.grad.data.zero_()
    b.grad.data.zero_()
  i+=1
  bar.progress(i)
  train_l = squared_loss(linreg(features, w, b), labels)
  st_chart.add_rows(np.array([0, train_l.mean().item()]))
  st.write('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

bar.progress(100)
print(true_w, '=', w)
print(true_b, '=', b)
true_w, '=', w
true_b, '=', b



# set_figsize()
# plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
# plt.show()
