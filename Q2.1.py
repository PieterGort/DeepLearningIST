import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt

# Load data from .npy files
train_images = np.load('octmnist/train_images.npy')
train_labels = np.load('octmnist/train_labels.npy')
test_images = np.load('octmnist/test_images.npy')
test_labels = np.load('octmnist/test_labels.npy')

# Make sure that the images are in the shape (num_samples, 28, 28, 1)
# If they are not, you will need to reshape them accordingly

# Normalize the images to the range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Convert the data and labels into TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(buffer_size=1024).batch(16)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(16)

class LogisticRegression(tf.keras.Model):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.flatten = Flatten(input_shape=(28, 28, 1))
        # Output layer for 4 classes with softmax activation
        self.dense_softmax = Dense(4, activation='softmax')

    def call(self, inputs):
        x = self.flatten(inputs)
        return self.dense_softmax(x)

# Initialize the model
model = LogisticRegression()

# Compile the model with SGD optimizer and binary cross-entropy loss
model.compile(optimizer=SGD(learning_rate=0.01),
            loss=SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Train the model
history = model.fit(train_dataset, epochs=20, validation_data=test_dataset)

# Evaluate on the test set
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Test accuracy: {test_accuracy}')

# Plot training & validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Plot training & validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
