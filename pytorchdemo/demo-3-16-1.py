import sys

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

st.title('实战Kaggle比赛：房价预测')

with st.echo():
  train_data = pd.read_csv('./data/kaggle_house/train.csv')
  test_data = pd.read_csv('./data/kaggle_house/test.csv')
  train_data.shape
  test_data.shape
  train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]]

'''#### 我们将所有的训练数据和测试数据的79个特征按样本连结'''
with st.echo():
  all_features = pd.concat([train_data.iloc[:, 1: -1], test_data.iloc[:, 1:]])
  all_features.shape
  all_features.dtypes
  all_features.size

'''#### 预处理数据
我们对连续数值的特征做标准化（standardization）：设该特征在整个数据集上的均值为μ，标准差为σ。那么，我们可以将该特征的每个值先减去μ再除以σ得到标准化后的每个特征值。对于缺失的特征值，我们将其替换成该特征的均值。'''
with st.echo():
  numberic_features = all_features.dtypes[all_features.dtypes != 'object'].index
  all_features.dtypes[all_features.dtypes != 'object']
  numberic_features
  all_features[numberic_features] = all_features[numberic_features].apply(
    lambda x: (x - x.mean()) / (x.std())
  )
  # 标准化后，每个特征的均值变为0，所以可以直接用0来替换缺失值
  all_features = all_features.fillna(0)
  # dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
  all_features = pd.get_dummies(all_features, dummy_na=True)
  all_features.shape
  all_features.dtypes
  all_features[0:2]

  n_train = train_data.shape[0]
  n_train
  train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
  test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
  train_labels_data = torch.tensor(train_data.SalePrice.values, dtype=torch.float)
  train_labels_data.shape
  train_labels = train_labels_data.view(-1,1)
  train_labels.shape

  # 训练模型
  loss = torch.nn.MSELoss()

  def get_net(feature_num):
    net = nn.Linear(feature_num, 1)
    for param in net.parameters():
      nn.init.normal_(param, mean=0, std=0.01)
    return net
'''
对数均方根误差
'''
st.text('$$\sqrt{\\frac{1}{n}\sum_{i=1}^n\left(\log(y_i)-\log(\hat y_i)\\right)^2}.$$')
with st.echo():
  def log_rmse(net, features, labels):
    with torch.no_grad():
      # 将小于1的值设成1，使得取对数时数值更稳定
      clipped_preds = torch.max(net(features), torch.tensor(1.0))
      rmse = torch.sqrt(2 * loss(clipped_preds.log(), labels.log()).mean())
    return rmse.item()

'''
不同在于使用了Adam优化算法。相对之前使用的小批量随机梯度下降，它对学习率相对不那么敏感
'''
with st.echo():
  def train(net, train_features, train_labels, test_features, test_labels,
            num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    # 这里使用了Adam优化算法
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    net = net.float()
    for epoch in range(num_epochs):
      for X, y in train_iter:
        l = loss(net(X.float()), y.float())
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
      train_ls.append(log_rmse(net, train_features, train_labels))
      if test_labels is not None:
        test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
'''
K 折交叉验证
'''
with st.echo():
  def get_k_fold_data(k, i, X, y):
    # 返回第i折交叉验证时所需要的训练和验证数据
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
      idx = slice(j * fold_size, (j + 1) * fold_size)
      X_part, y_part = X[idx, :], y[idx]
      if j == i:
        X_valid, y_valid = X_part, y_part
      elif X_train is None:
        X_train, y_train = X_part, y_part
      else:
        X_train = torch.cat((X_train, X_part), dim=0)
        y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid

  k = st.slider(label='K交叉验证数量', min_value=5, max_value=200, step=5, value=5)  # 学习率
  lr = st.slider(label='学习率', min_value=0.1, max_value=200.0, value=5.0)  # 学习率
  weight_decay = st.slider(label='权重衰减-权重', min_value=0, max_value=10, value=0)  # 学习率
  drop_prob1 = st.slider(label='丢弃--概率一层', min_value=0.0, max_value=1.0, value=0.2)  # 学习率
  drop_prob2 = st.slider(label='丢弃--概率二层', min_value=0.0, max_value=1.0, value=0.5)  # 学习率
  num_epochs = st.slider(label='迭代周期', min_value=100, max_value=1000, value=100, step=100)
  batch_size = st.slider(label='批量', min_value=64, max_value=640, value=64, step=64)
  loss = torch.nn.MSELoss()
  bar = st.progress(0)
  # st.subheader("损失折线图")
  # st_chart_loss = st.line_chart()
  # st.subheader("准确率图")
  # st_chart_tt = st.line_chart()

  def k_fold(k, X_train, y_train, num_epochs,
             learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum, timeStart = 0, 0, time.time()
    for i in range(k):
      data = get_k_fold_data(k, i, X_train, y_train)
      net = get_net(X_train.shape[1])
      train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                 weight_decay, batch_size)
      train_l_sum += train_ls[-1]
      valid_l_sum += valid_ls[-1]
      if i == 0:
        semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',
                     range(1, num_epochs + 1), valid_ls,
                     ['train', 'valid'])
      'fold %d, train rmse %f, valid rmse %f, time %.1f sec' % (i, train_ls[-1], valid_ls[-1], time.time() - timeStart)
    return train_l_sum / k, valid_l_sum / k

  train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
  '%d-fold validation: avg train rmse %f, avg valid rmse %f' % (k, train_l, valid_l)


'''### 预测并在Kaggle提交结果'''
# def train_and_pred(train_features, test_features, train_labels, test_data,
#                    num_epochs, lr, weight_decay, batch_size):
#   net = get_net(train_features.shape[1])
#   train_ls, _ = train(net, train_features, train_labels, None, None,
#                       num_epochs, lr, weight_decay, batch_size)
#   semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse')
#   'train rmse %f' % train_ls[-1]
#   preds = net(test_features).detach().numpy()
#   test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
#   submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
#   submission.to_csv('./submission.csv', index=False)
#
# train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)
