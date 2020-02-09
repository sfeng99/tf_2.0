import  tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class test:
    def __init__(self):
        (train_data, train_label), (_, _) = tf.keras.datasets.mnist.load_data()
        train_data = np.expand_dims(train_data.astype(np.float32) / 255.0, axis=-1)  # [60000, 28, 28, 1]
        self.mnist_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))


    def data_obj(self):
        X = tf.constant([2013,2014,2015,2016,2017])
        Y = tf.constant([12000, 14000, 15000, 16500, 17500])

        dataset = tf.data.Dataset.from_tensor_slices((X,Y))

        for x,y in dataset:
            print(x.numpy(),y.numpy())

    def imshow(self):
        for image, label in self.mnist_dataset:
            plt.title(label.numpy())
            plt.imshow(image.numpy()[:, :, 0])
            plt.show()
            break

    def rot90(self, image, label):
        image = tf.image.rot90(image)
        return image,label

    def im_rotate(self):
        self.mnist_dataset = self.mnist_dataset.map(self.rot90)
        for image, label in self.mnist_dataset:
            plt.title(label.numpy())
            plt.imshow(image.numpy()[:, :, 0])
            plt.show()
            break

    def data_batch(self):
        self.mnist_dataset = self.mnist_dataset.batch(4)

        for images, labels in self.mnist_dataset:  # image: [4, 28, 28, 1], labels: [4]
            fig, axs = plt.subplots(2, 2)
            for i in range(2):
                for j in range(2):
                    axs[i, j].set_title(labels.numpy()[i])
                    axs[i, j].imshow(images.numpy()[i, :, :, 0])
            plt.show()
            break

    def shuffle_data_batch(self):
        self.mnist_dataset = self.mnist_dataset.shuffle(buffer_size=10000).batch(4)

        for images, labels in self.mnist_dataset:
            fig, axs = plt.subplots(1, 4)
            for i in range(4):
                axs[i].set_title(labels.numpy()[i])
                axs[i].imshow(images.numpy()[i, :, :, 0])
            plt.show()



exp = test()
exp.data_batch()

