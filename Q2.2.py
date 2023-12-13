import tensorflow as tf
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.losses import SparseCategoricalCrossentropy
# import l2 regularization
from keras.regularizers import l2
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
epochs = 150
hidden_size = 200
batch_size = 256

# Convert the data and labels into TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(buffer_size=1024).batch(batch_size)
dev_dataset = tf.data.Dataset.from_tensor_slices((dev_images, dev_labels)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

class FeedforwardNetwork(tf.keras.Model):
    def __init__(self, hidden_size):
        super(FeedforwardNetwork, self).__init__()
        self.flatten = Flatten(input_shape=(28, 28, 1))
        self.dense1 = Dense(hidden_size, activation='relu')
        self.dense2 = Dense(hidden_size, activation='relu')
        self.output_layer = Dense(4, activation='softmax')

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)
# Define the neural network architecture with L2 regularization
class FeedforwardNetworkWithL2(tf.keras.Model):
    def __init__(self, hidden_size):
        super(FeedforwardNetworkWithL2, self).__init__()
        self.flatten = Flatten(input_shape=(28, 28, 1))
        # self.dense1 = Dense(hidden_size, activation='relu', kernel_regularizer=l2(l2_reg))
        # self.dense2 = Dense(hidden_size, activation='relu', kernel_regularizer=l2(l2_reg))
        self.dense1 = Dense(hidden_size, activation='relu')
        self.dense2 = Dense(hidden_size, activation='relu')
        self.output_layer = Dense(4, activation='softmax')

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)
    
# Define the neural network architecture with Dropout
class FeedforwardNetworkWithDropout(tf.keras.Model):
    def __init__(self, hidden_size, dropout_rate):
        super(FeedforwardNetworkWithDropout, self).__init__()
        self.flatten = Flatten(input_shape=(28, 28, 1))
        self.dense1 = Dense(hidden_size, activation='relu')
        self.dropout1 = Dropout(dropout_rate)
        self.dense2 = Dense(hidden_size, activation='relu')
        self.dropout2 = Dropout(dropout_rate)
        self.output_layer = Dense(4, activation='softmax')

    def call(self, inputs, training = False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout1(x, training = training)
        x = self.dense2(x)
        x = self.dropout2(x, training = training)
        return self.output_layer(x)

# Define different models to test
#model = FeedforwardNetwork(hidden_size=hidden_size)
model1 = FeedforwardNetworkWithL2(hidden_size=hidden_size)
model2 = FeedforwardNetworkWithDropout(hidden_size=hidden_size, dropout_rate=0.2)

# Compile the models
# L2 regularization (or weight decay) is directly applied at the optimizer level.
model1.compile(optimizer=SGD(learning_rate=0.1, decay=0.0001),
            loss=SparseCategoricalCrossentropy(),
            metrics=['accuracy'])

model2.compile(optimizer=SGD(learning_rate=0.1),
            loss=SparseCategoricalCrossentropy(),
            metrics=['accuracy'])


# Train the model
history1 = model1.fit(train_dataset, epochs=epochs, batch_size=batch_size, validation_data=dev_dataset)
history2 = model2.fit(train_dataset, epochs=epochs, batch_size=batch_size, validation_data=dev_dataset)

# Evaluate the model
test_loss1, test_accuracy1 = model1.evaluate(test_dataset, batch_size=batch_size)
print(f'Test accuracy model 1 with batch size {batch_size} and epochs {epochs}: {test_accuracy1}')
test_loss2, test_accuracy2 = model2.evaluate(test_dataset, batch_size=batch_size)
print(f'Test accuracy model 2 with batch size {batch_size} and epochs {epochs}: {test_accuracy2}')


plt.figure(figsize=(14, 5))
# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history1.history['accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.title(f'Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train1', 'Val1', 'Train2', 'Val2'], loc='upper left')

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title(f'Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train1', 'Val1', 'Train2', 'Val2'], loc='upper left')
plt.show()


    



    


