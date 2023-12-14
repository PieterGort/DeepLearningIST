import numpy as np
import tensorflow as tf
from keras.layers import Flatten, Dense
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
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(buffer_size=1024).batch(16)
dev_dataset = tf.data.Dataset.from_tensor_slices((dev_images, dev_labels)).batch(16)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(16)

num_classes = 4

class LogisticRegression(tf.keras.Model):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.flatten = Flatten(input_shape=(28, 28, 1))
        # Output layer for 4 classes with softmax activation
        self.dense_softmax = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.flatten(inputs)
        return self.dense_softmax(x)
    
learning_rates = [0.001, 0.01, 0.1]
best_accuracy = 0
best_lr = 0
history_dict = {}

for lr in learning_rates:
    # Initialize the model
    model = LogisticRegression()

    # Compile the model with SGD optimizer and binary cross-entropy loss
    model.compile(optimizer=SGD(learning_rate=lr),
                loss=SparseCategoricalCrossentropy(),
                metrics=['accuracy'])
    # Train the model
    history = model.fit(train_dataset, epochs=20, validation_data=dev_dataset)

    # Storing the history of each learning rate
    history_dict[lr] = history.history

    val_accuracy = max(history.history['val_accuracy'])
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_lr = lr
        best_model = model

print(f'best learning rate: {best_lr} with validation accuracy: {best_accuracy}')

# Evaluate on the test set
test_loss, test_accuracy = best_model.evaluate(test_dataset)
print(f'Test accuracy: {test_accuracy}')

plt.figure(figsize=(14, 5))
# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history_dict[best_lr]['accuracy'])
plt.plot(history_dict[best_lr]['val_accuracy'])
plt.title('Model accuracy for best LR')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history_dict[best_lr]['loss'])
plt.plot(history_dict[best_lr]['val_loss'])
plt.title('Model loss for best LR')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')

plt.show()
