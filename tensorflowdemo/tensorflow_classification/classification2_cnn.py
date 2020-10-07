# 分类学习
import tensorflow as tf
import numpy as np

# 下载库
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})  # 使用test数据，生成预测值，概率值：当前test数据的最大值的概率
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))  # 对比预测值和test数据真实数据
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 计算目前计算的有多少个对的，多少个错的
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})  # 计算一个正确率的百分比，越高表示越准确
    return result


# 权重变量 Weight变量
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 张量变量 biase变量
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 定义卷积 二维的卷积函数
def conv2d(x, W):
    '''
    定义卷积，tf.nn.conv2d函数是tensoflow里面的二维的卷积函数，
    x是图片的所有参数，W是此卷积层的权重，然后定义步长strides=[1,1,1,1]值，
    strides[0]和strides[3]的两个1是默认值，
    中间两个1代表padding时在x方向运动一步，y方向运动一步，padding采用的方式是SAME。
    :param x: 入参
    :param W: 权重
    :return:
    '''
    # stride [1, x_movement, y_movement, 1] 步幅 [1, x方向的移动, y方向的移动, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 定义池化pooling 池化 步幅的跨度增加
def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1] 步幅 [1, x方向的移动, y方向的移动, 1]
    # 接着定义池化pooling，为了得到更多的图片信息，padding时我们选的是一次一步，
    # 也就是strides[1]=strides[2]=1，这样得到的图片尺寸没有变化，而我们希望压缩一
    # 下图片也就是参数能少一些从而减小系统的复杂度，因此我们采用pooling来稀疏化参数，
    # 也就是卷积神经网络中所谓的下采样层。pooling 有两种，一种是最大值池化，一种是平均值池化，
    # 本例采用的是最大值池化tf.max_pool()。池化的核函数大小为2x2，因此ksize=[1,2,2,1]，步长为2，因此strides=[1,2,2,1]:
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# define placeholder for inputs to network 图片处理
# 利用占位符定义我们所需的神经网络的输入。
# tf.placeholder()就是代表占位符，这里的None代表无论输入有多少都可以，
# 因为输入只有一个特征，所以这里是1。
xs = tf.placeholder(tf.float32, [None, 784]) / 255  # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
# 我们还定义了dropout的placeholder，它是解决过拟合的有效手段
keep_prob = tf.placeholder(tf.float32)
# 接着呢，我们需要处理我们的xs，把xs的形状变成[-1,28,28,1]，-1代表先不考虑输入的图片例子多少这个维度，
# 后面的1是channel的数量，因为我们输入的图片是黑白的，因此channel是1，例如如果是RGB图像，那么channel就是3。
x_image = tf.reshape(xs, [-1, 28, 28, 1])
# print(x_image.shape)  # [n_samples, 28,28,1]


# 这一次我们一层层的加上了不同的 layer. 分别是:
# convolutional layer1 + max pooling; 建立卷积层1
## conv1 layer ##
# 接着我们定义第一层卷积,先定义本层的Weight,本层我们的卷积核patch的大小是5x5，因为黑白图片channel是1所以输入是1，输出是32个featuremap
W_conv1 = weight_variable([5, 5, 1, 32])  # patch 5x5, in size 1, out size 32 # 权重
# 接着定义bias，它的大小是32个长度，因此我们传入它的shape为[32]
b_conv1 = bias_variable([32])  # 张量
# 定义好了Weight和bias，我们就可以定义卷积神经网络的第一个卷积层h_conv1=conv2d(x_image,W_conv1)+b_conv1,同时我们对h_conv1进行非线性处理，
# 也就是激活函数来处理喽，这里我们用的是tf.nn.relu（修正线性单元）来处理，要注意的是，因为采用了SAME的padding方式，
# 输出图片的大小没有变化依然是28x28，只是厚度变厚了，因此现在的输出大小就变成了28x28x32
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 28x28x32 # 隐藏层数，加了非线性化的处理relu
# 最后我们再进行pooling的处理就ok啦，经过pooling的处理，输出大小就变为了14x14x32
h_pool1 = max_pool_2x2(h_conv1)  # output size 14x14x32

# convolutional layer2 + max pooling; 建立卷积层2
# 接着呢，同样的形式我们定义第二层卷积，本层我们的输入就是上一层的输出，本层我们的卷积核patch的大小是5x5，有32个featuremap所以输入就是32，输出呢我们定为64
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
# 接着我们就可以定义卷积神经网络的第二个卷积层，这时的输出的大小就是14x14x64
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# 最后也是一个pooling处理，输出大小为7x7x64
h_pool2 = max_pool_2x2(h_conv2)

# fully connected layer1 + dropout; 建立全连接层 + dropout
# 进入全连接层时, 我们通过tf.reshape()将h_pool2的输出值从一个三维的变为一维的数据, -1表示先不考虑输入图片例子维度, 将上一个输出结果展平.
# [n_samples,7,7,64]->>[n_samples,7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
# 此时weight_variable的shape输入就是第二个卷积层展平了的输出大小: 7x7x64， 后面的输出size我们继续扩大，定为1024
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
# 然后将展平后的h_pool2_flat与本层的W_fc1相乘（注意这个时候不是卷积了）
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# 如果我们考虑过拟合问题，可以加一个dropout的处理
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# fully connected layer2 to prediction.
# 接下来我们就可以进行最后一层的构建了，好激动啊, 输入是1024，最后的输出是10个 (因为mnist数据集就是[0-9]十个类)，prediction就是我们最后的预测值
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 开始预测
# loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))  # loss
# 学习效率要小于1
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
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
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
        if i % 50 == 0:
            # to see the step improvement
            # print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
            print(compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))
