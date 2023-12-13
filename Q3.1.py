import numpy as np
import matplotlib.pyplot as plt

hidden_size = 2
n_features = 3
n_classes = 1


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        # Initialize the weights of the hidden layer with a random normal and the bias
        
        self.W1 = np.random.normal(0.1, np.sqrt(0.01), (hidden_size, n_features))
        self.b1 = np.zeros(hidden_size)

        self.W2 = np.random.normal(0.1, np.sqrt(0.01), (n_classes, hidden_size))
        self.b2 = np.zeros(n_classes)        
    
    def sign(self, Z):
        return np.where(Z >= 0, 1, -1)
    
    def predict(self, X):
        Z1 = self.W1.dot(X) + self.b1
        h1 = self.sign(Z1)
        Z2 = self.W2.dot(h1) + self.b2
        h2 = self.sign(Z2)
        return h2
    
    def evaluate(self, X, y):
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        for x_i, y_i in zip(X, y):
            Z1 = self.W1.dot(x_i) + self.b1
            h1 = self.ReLu(Z1)
            Z2 = self.W2.dot(h1) + self.b2
            h2 = self.softmax(Z2)

            y_one_hot = np.eye(self.W2.shape[1])[y_i]
            dZ2 = h2 - y_one_hot
            dW2 = h1.T.dot(dZ2)
            db2 = np.sum(dZ2, axis=0)

            dh1 = self.W2.dot(dZ2.T)
            dZ1 = (Z1 > 0) * dh1.T
            dW1 = dZ1.T.dot(x_i)
            db1 = np.sum(dZ1, axis=0)

            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1

            # Compute the cross-entropy loss
            loss = self.cross_entropy(h2, y_one_hot, epsilon=1e-12)
            return loss
    
    def loss():
        pass