# tensorflow

    https://github.com/MorvanZhou/tutorials
    
    https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-01-classifier/

## 内容分布
1-11都属于Regression（回归）
20之后都是分类问题

这次我们会介绍如何使用TensorFlow解决Classification（分类）问题。 之前的视频讲解的是Regression (回归)问题。 分类和回归的区别在于输出变量的类型上。 通俗理解定量输出是回归，或者说是连续变量预测； 定性输出是分类，或者说是离散变量预测。如预测房价这是一个回归任务； 把东西分成几类, 比如猫狗猪牛，就是一个分类任务。
    
## 内部提供的激励函数
https://www.tensorflow.org/api_guides/python/nn
## 加速神经网络训练 (Speed Up Training)
大多数其他途径是在更新神经网络参数那一步上动动手脚

包括以下几种模式:
1. Stochastic Gradient Descent (SGD)
1. Momentum
1. AdaGrad
1. RMSProp
1. Adam 最优的方式结合力以上所有的方式的优点
## 优化器 提升学习效率
    各种 Optimizer 的对比 http://cs231n.github.io/neural-networks-3/
    
    最基础的GradientDescentOptimizer

## 分类问题
## 过拟合 overfitting
1. 产生原因 数据量过小
1. 解决方法
    1. 增大数据量
    1. 运用正规化. L1, l2 regularization等等
        这些方法适用于大多数的机器学习, 包括神经网络. 他们的做法大同小异, 我们简化机器学习的关键公式为 y=Wx . W为机器需要学习到的各种参数. 在过拟合中, W 的值往往变化得特别大或特别小. 为了不让W变化太大, 我们在计算误差上做些手脚. 原始的 cost 误差是这样计算, cost = 预测值-真实值的平方. 如果 W 变得太大, 我们就让 cost 也跟着变大, 变成一种惩罚机制. 所以我们把 W 自己考虑进来. 这里 abs 是绝对值. 这一种形式的 正规化, 叫做 l1 正规化. L2 正规化和 l1 类似, 只是绝对值换成了平方. 其他的l3, l4 也都是换成了立方和4次方等等. 形式类似. 用这些方法,我们就能保证让学出来的线条不会过于扭曲.
    1. dropout 一种专门用在神经网络的正规化的方法
        在训练的时候, 我们随机忽略掉一些神经元和神经联结 , 是这个神经网络变得”不完整”. 用一个不完整的神经网络训练一次.
        到第二次再随机忽略另一些, 变成另一个不完整的神经网络. 有了这些随机 drop 掉的规则, 我们可以想象其实每次训练的时候, 我们都让每一次预测结果都不会依赖于其中某部分特定的神经元. 像l1, l2正规化一样, 过度依赖的 W , 也就是训练参数的数值会很大, l1, l2会惩罚这些大的 参数. Dropout 的做法是从根本上让神经网络没机会过度依赖.
## 卷积神经网络 CNN (Convolutional Neural Network)
