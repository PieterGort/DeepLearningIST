import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt
import time

import utils

# Load OCT data
data = utils.load_oct_data()

train_images, train_labels = data["train"]
dev_images, dev_labels = data["dev"]
test_images, test_labels = data["test"]

# reshape the images to be 28x28
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
dev_images = dev_images.reshape(dev_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

# print shapes
print(train_images.shape)
print(test_images.shape)

# Load data from .npy files
# train_images = np.load('octmnist/train_images.npy')
# train_labels = np.load('octmnist/train_labels.npy')
# test_images = np.load('octmnist/test_images.npy')
# test_labels = np.load('octmnist/test_labels.npy')

# Make sure that the images are in the shape (num_samples, 28, 28, 1)
# If they are not, you will need to reshape them accordingly

# Normalize the images to the range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Convert the data and labels into TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(buffer_size=1024).batch(16)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(16)
# Define the neural network architecture
class FeedforwardNetwork(tf.keras.Model):
    def __init__(self, hidden_size):
        super(FeedforwardNetwork, self).__init__()
        # Define the layers
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28, 1))
        self.dense1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.output_layer = tf.keras.layers.Dense(4, activation='softmax')  # Assuming 4 classes

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)

# Load and preprocess your data
# ...

# Define hyperparameters
epochs = 20
learning_rate = 0.1
hidden_size = 200
batch_sizes = [16, 1024]

# Initialize the model
model = FeedforwardNetwork(hidden_size=hidden_size)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Train and evaluate the model using different batch sizes
for batch_size in batch_sizes:
    start_time = time.time()
    
    # Train the model
    history = model.fit(train_dataset, epochs=epochs, batch_size=batch_size, validation_data=test_dataset)
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_dataset, batch_size=batch_size)
    print(f'Test accuracy with batch size {batch_size}: {test_accuracy}')
    
    # Time taken
    end_time = time.time()
    print(f"Time taken for batch size {batch_size}: {end_time - start_time} seconds")
    
    # Plot training & validation accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Model accuracy for batch size {batch_size}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Plot training & validation loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model loss for batch size {batch_size}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
