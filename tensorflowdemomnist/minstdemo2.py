'''
https://www.jianshu.com/p/135c21e3db73
'''
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(dir(mnist))
print(mnist.train.num_examples)
print(mnist.validation.num_examples)
print(mnist.test.num_examples)

images = mnist.train.images
print((type(images),images.shape))
import matplotlib.pyplot as plt
import math
import numpy as np

def drawDigit(position, image, title):
  plt.subplot(*position)
  plt.imshow(image.reshape(-1, 28), cmap='gray_r')
  plt.axis('off')
  plt.title(title)

def batchDraw(batch_size):
  images,labels = mnist.train.next_batch(batch_size)
  image_number = images.shape[0]
  row_number = math.ceil(image_number ** 0.5)
  column_number = row_number
  plt.figure(figsize=(row_number, column_number))
  for i in range(row_number):
    for j in range(column_number):
      index = i * column_number + j
      if index < image_number:
        position = (row_number, column_number, index+1)
        image = images[index]
        title = 'actual:%d' %(np.argmax(labels[index]))
        drawDigit(position, image, title)

batchDraw(196)
plt.show()
