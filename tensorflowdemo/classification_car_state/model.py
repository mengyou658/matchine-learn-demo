import numpy as np
import tensorflow as tf

from tensorflowdemo.classification_car_state.data_processing import load_data, convert2onehot

data = load_data(download=False)
print(data)
new_data = convert2onehot(data)

print(new_data)
print(new_data.shape)


# 转换类型
new_data = new_data.values.astype(np.float32)

# 打乱顺序
np.random.shuffle(new_data)

sep = int(0.7 * len(new_data))
# 分成两部分：训练数据
train_data = new_data[:sep]
# 分成两部分：测试数据
test_data = new_data[sep:]

# 搭建网络
tf_input = tf.placeholder(tf.float32, [None, 25], "input")
tfx = tf_input[:, : 21]
tfy = tf_input[:, 21:]
