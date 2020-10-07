'''
https://www.jianshu.com/p/135c21e3db73
'''

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
batch_size = 100
X_holder = tf.placeholder(tf.float32)
y_holder = tf.placeholder(tf.float32)

Weights = tf.Variable(tf.zeros([784, 10]))
biases = tf.Variable(tf.zeros([1,10]))
predict_y = tf.nn.softmax(tf.matmul(X_holder, Weights) + biases)
# loss = tf.reduce_mean(-tf.reduce_sum(y_holder * tf.log(predict_y), 1))
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# train = optimizer.minimize(loss)

session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)
# 保存模型
m_saver = tf.train.Saver()
m_saver.restore(session, './model/mnist_slp-475')

import math
import matplotlib.pyplot as plt
import numpy as np

def drawDigit2(position, image, title, isTrue):
  plt.subplot(*position)
  plt.imshow(image.reshape(-1, 28), cmap='gray_r')
  plt.axis('off')
  if not isTrue:
    plt.title(title, color='red')
  else:
    plt.title(title)

def batchDraw2(batch_size):
  images,labels = mnist.test.next_batch(batch_size)
  predict_labels = session.run(predict_y, feed_dict={X_holder:images, y_holder:labels})
  image_number = images.shape[0]
  row_number = math.ceil(image_number ** 0.5)
  column_number = row_number
  plt.figure(figsize=(row_number+8, column_number+8))
  for i in range(row_number):
    for j in range(column_number):
      index = i * column_number + j
      if index < image_number:
        position = (row_number, column_number, index+1)
        image = images[index]
        actual = np.argmax(labels[index])
        predict = np.argmax(predict_labels[index])
        isTrue = actual==predict
        title = 'actual:%d\npredict:%d' %(actual,predict)
        drawDigit2(position, image, title, isTrue)

batchDraw2(100)
plt.show()
