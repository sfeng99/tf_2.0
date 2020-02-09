import tensorflow as tf
import numpy as np
import time

class MNISTLoarder():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (train_data, train_label), (test_data, test_label) = mnist.load_data()
        self.train_data = np.expand_dims(train_data.astype(np.float32)/255.,axis=-1)
        self.test_data = np.expand_dims(test_data.astype(np.float32)/255.,axis=-1)
        self.train_label = train_label.astype(np.int32)
        self.test_label = test_label.astype(np.int32)
        self.num_train_data = train_data.shape[0]
        self.num_test_data = test_data.shape[0]

    def get_batch(self, batch_size):
        index = np.random.randint(0, self.num_train_data, batch_size)
        return self.train_data[index, :], self.train_label[index]

class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units = 100, activation = tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units = 10)

    def call(self, input):
        x = self.flatten(input)
        x = self.dense1(x)
        x = self.dense2(x)
        output = tf.nn.softmax(x)
        return output
if __name__ == '__main__':
    n_epoch = 5
    batch_size = 64
    learning_rate = 1e-3

    model = MLP()
    checkpoint = tf.train.Checkpoint(Mymodel=model)
    manager = tf.train.CheckpointManager(checkpoint, directory='./save_MLP', max_to_keep=3)
    data_loader = MNISTLoarder()

    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    summary_writer = tf.summary.create_file_writer('./tensorboard_MLP')

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
        with summary_writer.as_default():
            tf.summary.scalar('loss',loss,step=batch_index)
        if batch_index % 1000 == 0:
            manager.save()
    time2 = time.time()
    print('time cost: %.1f seconds' % (time2-time1))

    # checkpoint.restore(tf.train.latest_checkpoint('./save_MLP'))
    # sparce_ctgrcal_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    # n_test_batches = int(data_loader.num_test_data // batch_size)
    # for batch_index in range(n_test_batches):
    #     start_index, end_index = batch_index*batch_size, (batch_index+1)*batch_size
    #     y_pred = model.predict(data_loader.test_data[start_index:end_index])
    #     sparce_ctgrcal_acc.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)
    # print('test accuracy: {}'.format(sparce_ctgrcal_acc.result()))