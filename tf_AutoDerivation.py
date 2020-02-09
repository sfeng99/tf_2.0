# tf 自动求导机制

import tensorflow as tf

x = tf.Variable(initial_value=3.)

with tf.GradientTape() as tape:
    y = tf.square(x)
y_grad = tape.gradient(y, x)

print(y_grad, y)