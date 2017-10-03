import tensorflow as tf

g = tf.Graph()  # Create a new graph
with g.as_default():
    c = tf.constant(10)
assert c.graph is g
indefaultgraph = tf.add(1, 2, name="indefaultgraph")
with g.as_default():
    ingraphg = tf.multiply(2, 3, name="ingraphg")
alsoindefaultgraph = tf.subtract(5, 1, name="alsoindefaultgraph")

# Using the context manager for session. Session close() is not needed in this case.
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./mygraph', sess.graph)
    writer.close()


