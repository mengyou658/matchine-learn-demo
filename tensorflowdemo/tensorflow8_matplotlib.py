# 例子3 添加层 def add_layer() 添加神经层
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# 在 Tensorflow 里定义一个添加层的函数可以很容易的添加神经层,为之后的添加省下不少时间.
# 神经层里常见的参数通常有weights、biases和激励函数。
def add_layer(inputs, in_size, out_size, activation_function=None):
    # 接下来，我们开始定义weights和biases。
    #
    # 因为在生成初始参数时，随机变量(normal distribution)会比全部为0要好很多，所以我们这里的weights为一个in_size行, out_size列的随机变量矩阵。
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # 在机器学习中，biases的推荐值不为0，所以我们这里是在0张量的基础上又加了0.1。
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

    # 下面，我们定义Wx_plus_b, 即神经网络未激活的值。其中，tf.matmul()是矩阵的乘法。
    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# 这次提到了怎样建造一个完整的神经网络,包括添加神经层,计算误差,训练步骤,判断是否在学习.

x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# 噪点 noise
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

# 利用占位符定义我们所需的神经网络的输入。
# tf.placeholder()就是代表占位符，这里的None代表无论输入有多少都可以，
# 因为输入只有一个特征，所以这里是1。
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# 输入层
layer1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# 输出层
prediction = add_layer(layer1, 10, 1, activation_function=None)

# 开始预测
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
# 学习效率要小于1
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# 初始化变量
init = tf.global_variables_initializer()
# 训练
with tf.Session() as sess:
    sess.run(init)
    # plot the real data
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x_data, y_data)
    plt.ion() #本次运行请注释，全局运行不要注释
    plt.show()
    # 比如这里，我们让机器学习1000次。机器学习的内容是train_step,
    # 用 Session 来 run 每一次 training 的数据，逐步提升神经网络的预测准确性。 (注意：当运算要用到placeholder时，就需要feed_dict这个字典来指定输入。)
    for i in range(1000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            # to see the step improvement
            # print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            # plot the prediction
            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
            plt.pause(0.1)
