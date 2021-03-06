"""
t_0 = 10 # Treated as 0-D tensor or "scalar"
t_1 = [1, 2, 3] # Treated as 1-D tensor or "vector"
t_2 = [[True, True, False] # Treated as 2-D tensor or "matrix"

Definition: NumPy is a computational package for manipulating n-dimensional array.
Tensors from NumPy arrays:
import numpy as np
t_0 = np.array(10, dtype=np.int32) # 0-D Tensor with 32-bit integer datatype.
t_1 = np.array([d"apple", d"banana", d"grape"]) # 1-D Tensors with string datatype. Note: Don't explicitly specify datatype when using string in NumPy.
t_2 = np.array([[True, False, False], [False, False, True], [False, True, False]], dtype=np.bool)
"""

import tensorflow as tf
import numpy as np

# Build a graph
nt = [4, 3]  # python native type
npt = np.array([4, 3], dtype=np.int32)  # Using numpy to create tensor
a = tf.constant(npt, name="constant_a")  # using nt as tensor
b = tf.reduce_sum(a, name="sum_b")
c = tf.reduce_prod(a, name="prod_c")
d = tf.multiply(b, c, name="mul_d")

# Launch the graph in a session
sess = tf.Session()
# Evaluate the tensor 'd'
print(sess.run(d))

# Write in tensorboard. Commenting as TENSORBOARD not installed.
writer = tf.summary.FileWriter('./mygraph', sess.graph)
writer.close()
sess.close()

