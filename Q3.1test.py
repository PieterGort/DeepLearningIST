import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.saving import get_custom_objects
from sklearn.model_selection import train_test_split
from keras import backend as K

# Custom sign activation function
def sign_activation(x):
    return K.sign(x)

# Get the custom object to make sure Keras recognizes the custom activation function
get_custom_objects()['sign_activation'] = sign_activation

# Constants
D = 2  # Number of features
num_samples = 10000  # Number of samples
hidden_size = 2  # Number of hidden units

# Generate the dataset
X = np.random.choice([-1, 1], size=(num_samples, D))
A, B = -1, 1  # Define your range
y = np.array([1 if A <= x.sum() <= B else -1 for x in X])

assert -D <= A <= B <= D, "A, B, D must satisfy -D <= A <= B <= D"

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define the MLP model
model = Sequential()
model.add(Dense(hidden_size, input_dim=D, activation='relu'))  # Hidden layer with tanh activation
model.add(Dense(1, activation='relu'))  # Output layer with custom sign activation

# Compile the model with Stochastic Gradient Descent (SGD) optimizer
model.compile(optimizer=SGD(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=16)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss:.3f}, Test accuracy: {accuracy:.3f}')
