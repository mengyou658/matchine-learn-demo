import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
import streamlit as st
import pandas as pd

from torch import nn


def load_data_fashion_mnist(batch_size):
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4

    mnist_train = torchvision.datasets.FashionMNIST(root='~/datasets/', train=True, download=True,
                                                    transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='~/datasets/', train=False, download=True,
                                                   transform=transforms.ToTensor())

    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter


def sgd(params, lr, batch_size):
    # 为了和原书保持一致，这里除以了batch_size，但是应该是不用除的，因为一般用PyTorch计算loss时就默认已经
    # 沿batch维求了平均了。
    for param in params:
        param.data -= lr * param.grad / batch_size  # 注意这里更改param时用的param.data


def getDevice(gpuid=0):
    if torch.cuda.is_available():
        device = torch.device("cuda:%d" % gpuid)
    else:
        device = torch.device("cpu")
        print("CUDA is not available, fall back to CPU.")
    return device


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None, figsize=(3.5, 2.5)):
    fit1 = plt.figure()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    st.pyplot(fit1)


def linreg(X, w, b):
    return torch.mm(X, w) + b


def squared_loss(y_hat, y):
    # 注意这里返回的是向量, 另外, pytorch里的MSELoss并没有除以 2
    return ((y_hat - y.view(y_hat.size())) ** 2) / 2


# def evaluate_accuracy(data_iter, net):
#   acc_sum, n = 0.0, 0
#   for X, y in data_iter:
#     if isinstance(net, torch.nn.Module):
#       net.eval() # 评估模式, 这会关闭dropout
#       acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
#       net.train() # 改回训练模式
#     else: # 自定义的模型
#       if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
#         # 将is_training设置成False
#         acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
#       else:
#         acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
#     n += y.shape[0]
#   return acc_sum / n

def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()  # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()  # 改回训练模式
            else:  # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if ('is_training' in net.__code__.co_varnames):  # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


def train_ch3(net, train_iter, test_iter, loss,
              num_epochs, batch_size, params=None, lr=None,
              optimizer=None, device=None
              , bar=None, st_chart_loss=None, st_chart_tt=None, ):
    process = 0
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

        if bar:
            process += 1
            if process > 100:
                process = 100
            bar.progress(process)
            test_acc = evaluate_accuracy(test_iter, net)
            st.text('epoch %d, loss %.4f, train acc %.3f, test acc %.3f , time %.1f sec' % (
                epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc, time.time() - timeStart))
            if st_chart_loss:
                st_chart_loss.add_rows(pd.DataFrame([[float('%.4f' % (train_l_sum / n))]], columns=['loss']))
            if st_chart_tt:
                st_chart_tt.add_rows(pd.DataFrame([[float('%.3f' % (train_acc_sum / n)), float('%.3f' % (test_acc))]],
                                                  columns=['train acc', 'test acc']))


def load_data_fashion_mnist(batch_size, resize=None, num_workers=0, root='~/datasets'):
    """Download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter


def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    device,
    loss = torch.nn.CrossEntropyLoss()
    print("start train:")
    bar = st.progress(0)
    total = len(train_iter) * num_epochs
    countTotal = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            startTmp = time.time()
            # print("every start:", batch_count)
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
            batch_count += 1
            countTotal += 1
            bar.progress(countTotal / total)
        # print("epoch end:", batch_count, (time.time() - startTmp))
        test_acc = evaluate_accuracy(test_iter, net)
        res = 'epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec' % (
            epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start)
        print(res)
        st.write(res)
    print("end train:")
