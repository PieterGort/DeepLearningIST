import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.losses import SparseCategoricalCrossentropy
from keras.regularizers import l2
import matplotlib.pyplot as plt
import utils

class FeedforwardNetwork:
    def __init__(self, input_shape, num_classes, hidden_size=200, number_of_layers=2, dropout_rate=0.2, 
                 activation_function='relu', learning_rate=0.1, l2_regularization=0):
        super(FeedforwardNetwork, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.number_of_layers = number_of_layers
        self.dropout_rate = dropout_rate
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.l2_regularization = l2_regularization
        self.optimizer = SGD(learning_rate=self.learning_rate)
        self.model = self.build_model()
    
    def tfdataset(self, images, labels, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
        return dataset
    
    def build_model(self):
        model = Sequential()
        # Flatten the input to be one-dimensional array
        model.add(Flatten(input_shape=self.input_shape))
        # Add layers to the model
        model.add(Dense(self.hidden_size, activation=self.activation_function, 
                        kernel_regularizer=l2(self.l2_regularization)))  # First hidden layer
        if self.dropout_rate > 0.0:
            model.add(Dropout(self.dropout_rate))
        for _ in range(self.number_of_layers - 1):  # Additional hidden layers
            model.add(Dense(self.hidden_size, activation=self.activation_function, 
                            kernel_regularizer=l2(self.l2_regularization)))
        # Output layer: assuming it is a multi-class classification problem
        model.add(Dense(self.num_classes, activation='softmax'))
        # Compile the model
        model.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def summary(self):
        self.model.summary()

    def compile_model(self):
        self.model.compile(optimizer=SGD(learning_rate=self.learning_rate),
                           loss=SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])

    def train(self, train_images, train_labels, dev_images, dev_labels, batch_size, epochs):
        train_dataset = self.tfdataset(train_images, train_labels, batch_size)
        dev_dataset = self.tfdataset(dev_images, dev_labels, batch_size)
        self.compile_model()  # Ensure the model is compiled
        history = self.model.fit(train_dataset, epochs=epochs, validation_data=dev_dataset)
        return history
    
    def evaluate(self, test_images, test_labels, batch_size):
        test_dataset = self.tfdataset(test_images, test_labels, batch_size)
        test_loss, test_accuracy = self.model.evaluate(test_dataset)
        print(f'Test accuracy: {test_accuracy}')
        return test_loss, test_accuracy

    def plot_history(self, history):
        plt.figure(figsize=(14, 5))
        # Plot training & validation accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f'Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')

        # Plot training & validation loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.show()

# Define the parameters from the figure
input_shape = (28, 28, 1)  # Input shape for each image
num_classes = 4  # Number of classes

# Load OCT data
data = utils.load_oct_data()

train_images, train_labels = data["train"]
dev_images, dev_labels = data["dev"]
test_images, test_labels = data["test"]

# reshape the images to be 28x28
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
dev_images = dev_images.reshape(dev_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

# Normalize the images to the range [0, 1]
train_images = train_images / 255.0
dev_images = dev_images / 255.0
test_images = test_images / 255.0

# Define the parameters from the figure
input_shape = (28, 28, 1)  # Input shape for each image
num_classes = 4  # Number of classes
bs = 256
eps = 150

# Create an instance of the FeedforwardNetwork
network = FeedforwardNetwork(input_shape=input_shape, num_classes=num_classes)

# Print the model summary
network.summary()

# Create TensorFlow datasets for training, validation and testing
train_dataset = network.tfdataset(train_images, train_labels, batch_size=bs)
dev_dataset = network.tfdataset(dev_images, dev_labels, batch_size=bs)
test_dataset = network.tfdataset(test_images, test_labels, batch_size=bs)

# Train the model with the training data
history = network.train(train_images, train_labels, dev_images, dev_labels, batch_size=bs, epochs=eps)

# Evaluate the model with the test data
test_loss, test_accuracy = network.evaluate(test_images, test_labels, batch_size=bs)

# Plot the training history
network.plot_history(history)