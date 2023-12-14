import numpy as np
import tensorflow as tf
from keras.layers import Flatten, Dense
from keras.models import Sequential
from keras.optimizers import SGD
# import logisticregression from keras
from keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt
import utils

# Load OCT data
data = utils.load_oct_data()

train_images, train_labels = data["train"]
dev_images, dev_labels = data["dev"]
test_images, test_labels = data["test"]

print(train_labels[0:10])

# reshape the images to be 28x28
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
dev_images = dev_images.reshape(dev_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

# print shapes
print(train_images.shape)
print(dev_images.shape)
print(test_images.shape)

# Normalize the images to the range [0, 1]
train_images = train_images / 255.0
dev_images = dev_images / 255.0
test_images = test_images / 255.0

# Convert the data and labels into TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(16)
dev_dataset = tf.data.Dataset.from_tensor_slices((dev_images, dev_labels)).batch(16)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(16)

num_classes = 4  # Assuming 4 classes for multi-class classification

# Initialize the model for multi-class classification
model = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(num_classes, activation='softmax')  # One neuron per class
])

# Compile the model with SGD optimizer and sparse categorical cross-entropy loss
model.compile(optimizer=SGD(learning_rate=0.1),
              loss=SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Train the model
history = model.fit(x=train_images, y=train_labels, validation_data=dev_dataset, epochs=20, shuffle=True)

# Evaluate on the test set
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Test accuracy: {test_accuracy}')

plt.figure(figsize=(14, 5))
# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy per epoch')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Validation accuracy'], loc='upper left')

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.title('Model loss per epoch')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training loss'], loc='upper left')

plt.show()
