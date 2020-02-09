# Running on Google colab
# time cost: 91.3s
# test_acc:  0.99

import tensorflow as tf
from tf_keras_MLP import MNISTLoarder
import time

class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters = 32,
            kernel_size = [5,5],
            padding = 'same',
            activation = tf.nn.relu
        )
        self.pool1 = tf.keras.layers.MaxPool2D(
            pool_size = [2,2],
            strides = 2
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPool2D(
            pool_size=[2, 2],
            strides=2
        )
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.conv1(inputs)  # [batch_size, 28, 28, 32]
        x = self.pool1(x)  # [batch_size, 14, 14, 32]
        x = self.conv2(x)  # [batch_size, 14, 14, 64]
        x = self.pool2(x)  # [batch_size, 7, 7, 64]
        x = self.flatten(x)  # [batch_size, 7 * 7 * 64]
        x = self.dense1(x)  # [batch_size, 1024]
        x = self.dense2(x)  # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output

if __name__ == '__main__':
    n_epoch = 5
    batch_size = 64
    learning_rate = 1e-3

    model = CNN()
    data_loader = MNISTLoarder()
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

    n_batches = int(data_loader.num_train_data // batch_size * n_epoch)
    print('number_of_batches:{}'.format(n_batches))

    time1 = time.time()
    for batch_index in range(n_batches):
        X, y = data_loader.get_batch(batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true = y, y_pred = y_pred)
            loss = tf.reduce_mean(loss)
            if batch_index % 100 ==0:
                print("batch %d: loss %f" % (batch_index, loss.numpy()))
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars = zip(grads, model.variables))
    time2 = time.time()
    print('time cost: %.1f seconds' % (time2-time1))

    sparce_ctgrcal_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    n_test_batches = int(data_loader.num_test_data // batch_size)
    for batch_index in range(n_test_batches):
        start_index, end_index = batch_index*batch_size, (batch_index+1)*batch_size
        y_pred = model.predict(data_loader.test_data[start_index:end_index])
        sparce_ctgrcal_acc.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)
    print('test accuracy: {}'.format(sparce_ctgrcal_acc.result()))