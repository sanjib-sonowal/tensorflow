"""
Tensorflow Self Tutorial.
Content taken from https://www.tensorflow.org/
"""

import tensorflow as tf

# Create two floating point Tensors node1 and node2 as follows.
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)  # also tf.float32 implicitly
print(node1, node2)

# To actually evaluate the nodes, we must run the computational graph within a session.
# A session encapsulates the control and state of the TensorFlow runtime.
sess = tf.Session()
print(sess.run([node1, node2]))

# We can build more complicated computations by combining Tensor nodes with operations (Operations are also nodes).
node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))

# A graph can be parameterized to accept external inputs, known as placeholders
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

# We can make the computational graph more complex by adding another operation
add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b: 4.5}))

# To make the model trainable, we need to be able to modify the graph to get new outputs with the same input.
# Variables allow us to add trainable parameters to a graph
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
# Constants are initialized when you call tf.constant, and their value can never change.
# By contrast, variables are not initialized when you call tf.Variable.
# To initialize all the variables in a TensorFlow program, you must explicitly call a special operation as below.
init = tf.global_variables_initializer()
sess.run(init)




