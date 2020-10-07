# Placeholder 传入值
# 这一次我们会讲到 Tensorflow 中的 placeholder ,
# placeholder 是 Tensorflow 中的占位符，暂时储存变量
# Tensorflow 如果想要从外部传入data, 那就需要用到 tf.placeholder(),
# 然后以这种形式传输数据 sess.run(***, feed_dict={input: **}).
import tensorflow as tf

#在 Tensorflow 中需要定义 placeholder 的 type ，一般为 float32 形式
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

# mul = multiply 是将input1和input2 做乘法运算，并输出为 output
output = tf.multiply(input1, input2)

# 接下来, 传值的工作交给了 sess.run() , 需要传入的值放在了feed_dict={}
# 并一一对应每一个 input.
# placeholder 与 feed_dict={} 是绑定在一起出现的。
with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1:[7.], input2: [2.]}))
