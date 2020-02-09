import tensorflow as tf

# rewrite __init__, build, call
class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(name='w', shape=[input_shape[-1],self.units], initializer=tf.random_normal_initializer())
        #add_variable was depreciated
        self.b = self.add_weight(name='b', shape=[self.units], initializer=tf.random_normal_initializer())

    def call(self, inputs):
        y_pred = tf.matmul(inputs, self.w) + self.b
        return y_pred

class LinearModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer = LinearLayer(1)

    def call(self, inputs):
        outputs = self.layer(inputs)
        return outputs

X = tf.constant([[1.,2.,3.],[4.,5.,6.]])
y = tf.constant([10.,20.])
model = LinearModel()
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)


for i in range(10):
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.reduce_mean(tf.square(y_pred-y))
    grads = tape.gradient(loss,model.variables)
    optimizer.apply_gradients(grads_and_vars = zip(grads, model.variables))

print(model.variables)
