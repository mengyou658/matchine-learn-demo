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

image = mnist.train.images[1].reshape(-1, 28)
plt.subplot(131)
plt.imshow(image)
plt.axis('off')
plt.subplot(132)
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.subplot(133)
plt.imshow(image, cmap='gray_r')
plt.axis('off')
plt.show()
