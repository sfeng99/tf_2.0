import tensorflow as tf
import numpy as np

X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
Y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

X = (X_raw - X_raw.min())/(X_raw.max() - X_raw.min())
Y = (Y_raw - Y_raw.min())/(Y_raw.max() - Y_raw.min())

X = tf.constant(X)
y = tf.constant(Y)
a = tf.Variable(initial_value = 0.)
b = tf.Variable(initial_value = 0.)
variables = [a, b]
n_epoch = 1000
optimizer = tf.keras.optimizers.SGD(learning_rate = 1e-3)

for e in range(n_epoch):
    with tf.GradientTape() as tape:
        y_pred = a * X + b
        loss = 0.5 * tf.reduce_sum(tf.square(y_pred - y))
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(grads_and_vars = zip(grads, variables))

print(a.numpy(), b.numpy())