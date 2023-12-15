#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neural_network import MLPClassifier as MLPsklearn
import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        
        pred_y = self.predict(x_i)
        if pred_y != y_i:
            self.W[y_i] += x_i
            self.W[pred_y] -= x_i

class LogisticRegression(LinearModel):
        def update_weight(self, x_i, y_i, learning_rate=0.001):

            y_hat = np.dot(self.W, x_i)
            y_hat -=np.max(y_hat)
            y_probabilities = np.exp(y_hat / np.sum(np.exp(y_hat)))

            y_one_hot = np.zeros(self.W.shape[0])
            y_one_hot[y_i] = 1

            # SGD update
            gradient = np.outer(y_probabilities - y_one_hot, x_i)
            self.W -= learning_rate * gradient

class MLP(object):
    def __init__(self, n_classes, n_features, hidden_size):
        # Xavier/Glorot initialization for weights
        self.W1 = np.random.randn(hidden_size, n_features) * np.sqrt(2. / n_features)
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(n_classes, hidden_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((n_classes, 1))
        
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        return Z > 0
    
    def softmax(self, Z):
        e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return e_Z / np.sum(e_Z, axis=0, keepdims=True)
    
    def cross_entropy(self, Y_hat, Y):
        m = Y.shape[1]
        Y_hat = Y_hat.clip(min=1e-10, max=1 - 1e-10)  # Clip values to avoid log(0)
        return -np.sum(Y * np.log(Y_hat)) / m
    
    def predict(self, X):
        A1 = self.relu(np.dot(self.W1, X.T) + self.b1)
        A2 = self.softmax(np.dot(self.W2, A1) + self.b2)
        return np.argmax(A2, axis=0)
    
    def evaluate(self, X, y):
        y_hat = self.predict(X)
        return np.mean(y_hat == y)
    
    def train_epoch(self, X, y, learning_rate=0.001):
        losses = []
        m = X.shape[0]
        
        for i in range(m):
            x_i = X[i].reshape(-1, 1)  # Reshape x_i to (n_features, 1)
            y_i = y[i]
            
            # Forward pass
            Z1 = np.dot(self.W1, x_i) + self.b1
            A1 = self.relu(Z1)
            Z2 = np.dot(self.W2, A1) + self.b2
            A2 = self.softmax(Z2)

            # Convert label to one-hot vector
            y_one_hot = np.zeros((self.W2.shape[0], 1))
            y_one_hot[y_i, 0] = 1

            # Loss calculation
            loss = self.cross_entropy(A2, y_one_hot)
            losses.append(loss)

            # Backward pass
            dZ2 = A2 - y_one_hot
            dW2 = dZ2.dot(A1.T)
            db2 = dZ2

            dA1 = self.W2.T.dot(dZ2)
            dZ1 = dA1 * self.relu_derivative(Z1)
            dW1 = dZ1.dot(x_i.T)
            db1 = dZ1

            # Update parameters
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1

        return np.mean(losses)

def plot(epochs, train_accs, val_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    plt.show()

def plot_loss(epochs, loss):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_oct_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # print("train_X shape: {}".format(train_X.shape))
    # print("train_y shape: {}".format(train_y.shape))
    # print(f'There are {train_X.shape[0]} observations with {n_feats} features classified into {n_classes} classes.')

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    valid_accs = []
    train_accs = []
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))
    
    # Compare the results from the written code with sklearn package for LogisticRegression
    # clf = LR(fit_intercept=False, penalty='none')
    # clf.fit(train_X, train_y)
    # print(clf.score(train_X, train_y))
    # print(clf.score(dev_X, dev_y))

    # compare MLP results with sklearn MLP package
    # clf = MLPsklearn(hidden_layer_sizes=(opt.hidden_size,), max_iter=opt.epochs, learning_rate_init=opt.learning_rate, random_state=42)
    # clf.fit(train_X, train_y)
    # print("The final test accuracy for the MLP from the scikit-learn package is:", clf.score(test_X, test_y))

    # plot
    plot(epochs, train_accs, valid_accs)
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss)


if __name__ == '__main__':
    main()
