import tensorflow as tf
import numpy as np

a = tf.placeholder(tf.int32, shape=[2], name="input")
b = tf.reduce_prod(a, name="prod_b")
c = tf.reduce_sum(a, name="sum_c")
d = tf.add(b, c, name="add_d")

sess = tf.Session()
input_dict = {a: np.array([1, 2], dtype=np.int32)}
# Fetch the value of 'd' by feeding the values of 'input_vector' into 'a'
print(sess.run(d, feed_dict=input_dict))
sess.close()
