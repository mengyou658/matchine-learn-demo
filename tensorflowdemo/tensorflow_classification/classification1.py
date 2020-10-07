# 分类学习
import tensorflow as tf
import numpy as np

# 下载库
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
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


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs}) # 使用test数据，生成预测值，概率值：当前test数据的最大值的概率
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1)) # 对比预测值和test数据真实数据
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # 计算目前计算的有多少个对的，多少个错的
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys}) # 计算一个正确率的百分比，越高表示越准确
    return result

# 这次提到了怎样建造一个完整的神经网络,包括添加神经层,计算误差,训练步骤,判断是否在学习.

# x_data = np.linspace(-1,1,300)[:, np.newaxis]
# noise = np.random.normal(0, 0.05, x_data.shape)
# y_data = np.square(x_data) - 0.5 + noise

# 噪点 noise
# noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
# y_data = np.square(x_data) - 0.5 + noise

# 利用占位符定义我们所需的神经网络的输入。
# tf.placeholder()就是代表占位符，这里的None代表无论输入有多少都可以，
# 因为输入只有一个特征，所以这里是1。
xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
ys = tf.placeholder(tf.float32, [None, 10])

# 输入层
# layer1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# 输出层
# prediction = add_layer(layer1, 10, 1, activation_function=None)
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

# 开始预测
# loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
# 学习效率要小于1
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)
# 初始化变量
init = tf.global_variables_initializer()
# 训练
with tf.Session() as sess:
    sess.run(init)
    # 比如这里，我们让机器学习1000次。机器学习的内容是train_step,
    # 用 Session 来 run 每一次 training 的数据，逐步提升神经网络的预测准确性。 (注意：当运算要用到placeholder时，就需要feed_dict这个字典来指定输入。)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        # sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
        if i % 50 == 0:
            # to see the step improvement
            # print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
            print(compute_accuracy(mnist.test.images, mnist.test.labels))
