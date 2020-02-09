import tensorflow as tf
from tf_keras_MLP import MNISTLoarder

inputs = tf.keras.Input(shape=(28,28,1))
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)(x)
x = tf.keras.layers.Dense(units=10)(x)
outputs = tf.keras.layers.Softmax()(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3),
    loss = tf.keras.losses.sparse_categorical_crossentropy,
    metric =[tf.keras.metrics.SparseCategoricalAccuracy]
)

dataloader = MNISTLoarder()
num_epochs = 5
batch_size = 50

model.fit(
    dataloader.train_data,
    dataloader.train_label,
    epochs = num_epochs,
    batch_size = batch_size
)

sparce_ctgrcal_acc = tf.keras.metrics.SparseCategoricalAccuracy()
n_test_batches = int(dataloader.num_test_data // batch_size)
for batch_index in range(n_test_batches):
    start_index, end_index = batch_index*batch_size, (batch_index+1)*batch_size
    y_pred = model.predict(dataloader.test_data[start_index:end_index])
    sparce_ctgrcal_acc.update_state(y_true=dataloader.test_label[start_index: end_index], y_pred=y_pred)
print('test accuracy: {}'.format(sparce_ctgrcal_acc.result()))

