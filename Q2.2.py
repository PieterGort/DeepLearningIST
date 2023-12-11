import tensorflow as tf
from keras.layers import Flatten, Dense
from keras.optimizers import SGD
from keras.losses import SparseCategoricalCrossentropy
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
print(dev_images.shape)
print(test_images.shape)

# Normalize the images to the range [0, 1]
train_images = train_images / 255.0
dev_images = dev_images / 255.0
test_images = test_images / 255.0

# Define hyperparameters
epochs = 20
learning_rates = [1, 0.1, 0.01, 0.001]
hidden_size = 200
batch_sizes = [16, 1024]

# Define the neural network architecture
class FeedforwardNetwork(tf.keras.Model):
    def __init__(self, hidden_size):
        super(FeedforwardNetwork, self).__init__()
        # Define the layers
        self.flatten = Flatten(input_shape=(28, 28, 1))
        self.dense1 = Dense(hidden_size, activation='relu')
        self.dense2 = Dense(hidden_size, activation='relu')
        self.output_layer = Dense(4, activation='softmax')  # Assuming 4 classes

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)
    
best_batch = 0
best_accuracy = 0
history_dict = {}

# Train and evaluate the model using different batch sizes
for batch_size in batch_sizes:
    start_time = time.time()

    # Convert the data and labels into TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(buffer_size=1024).batch(batch_size)
    dev_dataset = tf.data.Dataset.from_tensor_slices((dev_images, dev_labels)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

    # Initialize the model
    model = FeedforwardNetwork(hidden_size=hidden_size)

    # Compile the model
    model.compile(optimizer=SGD(learning_rate=0.1),
                loss=SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

    # Train the model
    history = model.fit(train_dataset, epochs=epochs, batch_size=batch_size, validation_data=dev_dataset)

    history_dict[batch_size] = history.history

    val_accuracy = max(history.history['val_accuracy'])
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_batch = batch_size
        best_model = model

    # Time taken per batch size
    end_time = time.time()
    plt.figure(figsize=(14, 5))
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Model accuracy for batch size {batch_size}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model loss for batch size {batch_size}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
    
print(f'best learning rate: {best_batch} with validation accuracy: {best_accuracy}')
    
# Evaluate the model
test_loss, test_accuracy = best_model.evaluate(test_dataset, batch_size=best_batch)
print(f'Test accuracy with batch size {best_batch}: {test_accuracy}')


    


