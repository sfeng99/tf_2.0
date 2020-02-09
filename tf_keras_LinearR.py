import tensorflow as tf


X = tf.constant([[1., 2., 3.], [4., 5., 6.]])
y = tf.constant([[10.],[20.]])
n_epoch  = 100

class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            units = 1,
            activation = None,
            kernel_initializer = tf.zeros_initializer(),
            bias_initializer = tf.zeros_initializer()
        )

    def call(self, input):
        output = self.dense(input)
        return output

model = Linear()
optimizer = tf.keras.optimizers.SGD(learning_rate = 1e-3)

for e in range(n_epoch):
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.reduce_sum(tf.square(y_pred - y))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars = zip(grads, model.variables))
print(model.variables)

