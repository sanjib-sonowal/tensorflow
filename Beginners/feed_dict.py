import tensorflow as tf
import numpy as np

# Example-1
a = tf.add(2, 3)
b = tf.multiply(a, 4)

sess = tf.Session()
replace_dist = {a: 15}
print(sess.run(b, replace_dist))
sess.close()

# Example-2
x = tf.placeholder(tf.float32, shape=(1024, 1024))
y = tf.matmul(x, x)  # Matrix Multiplication

with tf.Session() as sess:
    rand_array = np.random.rand(1024, 1024)
    print(sess.run(y, feed_dict={x: rand_array}))
    sess.close()

