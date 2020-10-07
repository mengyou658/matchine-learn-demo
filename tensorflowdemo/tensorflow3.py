# 这节课我们学习如何在 Tensorflow 中使用 Variable
# 定义语法： state = tf.Variable()
import tensorflow as tf

state = tf.Variable(0, name="counter")
# 定义常量 one
one = tf.constant(1)
# 定义加法步骤（注意：此步骤没有直接计算）
new_value = tf.add(state, one)
# 将 state 更新为 new_value
update = tf.assign(state, new_value)

# 如果你在 Tensorflow 中设定了变量，那么初始化变量是最重要的！！
# 所以定义了变量以后, 一定要定义tf.global_variables_initializer()

# 如果定义 Variable, 就一定要 initialize
# init = tf.initialize_all_variables() # tf 马上就要废弃这种写法
init = tf.global_variables_initializer()  # 替换成这样就好

# 使用session
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
