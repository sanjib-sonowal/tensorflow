import tensorflow as tf

a = tf.constant(4, name="input_a")
b = tf.constant(3, name="input_b")
c = tf.add(a, b, "add_c")
d = tf.multiply(a, b, "mul_d")
e = tf.multiply(c, d, "mul_e")

sess = tf.Session()
print(sess.run(e))
sess.close()



