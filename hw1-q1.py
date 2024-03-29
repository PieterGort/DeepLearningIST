#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as LR
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
    def __init__(self, n_classes, n_features, **kwargs):
        super(Perceptron, self).__init__(n_classes, n_features)
        self.learning_rate = kwargs.get('learning_rate', 1)
        self.epochs= kwargs.get('epochs', 20)
        self.W = np.zeros((n_classes, n_features + 1))

    def train_epoch(self, X, y, learning_rate):
        n_samples = np.shape(X)[0]
        X = np.concatenate((np.ones((n_samples, 1)), X), axis=1)
        mistakes = 0
        for x_i, y_i in zip(X, y):
            y_hat = np.argmax(self.W.dot(x_i))
            if y_hat != y_i:
                mistakes += 1
                self.W[y_i, :] += learning_rate * x_i
                self.W[y_hat, :] -= learning_rate * x_i
        return self.W
    
    def predict(self, X):
        n_samples = np.shape(X)[0]
        X = np.concatenate((np.ones((n_samples, 1)), X), axis=1)
        predicted_labels = []
        for x_i in X:
            y_hat = np.argmax(self.W.dot(x_i))
            predicted_labels.append(y_hat)
        predicted_labels = np.array(predicted_labels)
        return predicted_labels

    def evaluate(self, X, y):
        accuracy = np.mean(self.predict(X) == y)
        return accuracy


class LogisticRegression(LinearModel):
    def __init__(self, n_classes, n_features):
        super(LogisticRegression, self).__init__(n_classes, n_features)
        self.W = self.init_weights(n_features, n_classes)

    def init_weights(self, n_features, n_classes):
        t = np.sqrt(6 / (n_features + n_classes))
        weights = np.random.uniform(-t, t, (n_classes, n_features + 1))
        return weights
     
    def train_epoch(self, X, y, learning_rate):
        n, p = np.shape(X)
        
        # # add bias
        X = np.concatenate((np.ones((n, 1)), X), axis=1)

        for x_i, y_i in zip(X, y):

            # probability scores according to the model (n_classes x 1)
            y_label_scores = np.expand_dims((self.W).dot(x_i), axis=1)

            # one-hot encoding of the true label (n_labels x 1)
            y_one_hot = np.zeros((np.size(self.W, 0), 1))
            y_one_hot[y_i] = 1

            # softmax function
            label_probabilities = np.exp(y_label_scores) / np.sum(np.exp(y_label_scores))

            # SGD update
            self.W = self.W + learning_rate * (y_one_hot - label_probabilities).dot(np.expand_dims(x_i, axis = 1).T)
        return self.W
    
    def predict(self, X):
        n, p = np.shape(X)
        X = np.concatenate((np.ones((n, 1)), X), axis=1)
        y_hat = np.argmax(X.dot(self.W.T), axis=1)
        return y_hat

    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
    

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
    
    def ReLu(self, Z):
        return np.maximum(0, Z)
    
    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, keepdims=True))
        return exp_Z / np.sum(exp_Z, keepdims=True)
    
    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.

        Z1 = X.dot(self.W1.T) + self.b1
        h1 = self.ReLu(Z1)
        Z2 = h1.dot(self.W2) + self.b2
        h2 = self.softmax(Z2)

        return np.argmax()
    
    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        """
        Dont forget to return the loss of the epoch.
        """
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
    
    def cross_entropy(self, predictions, targets, epsilon=1e-12):
        predictions = np.clip(predictions, epsilon, 1. - epsilon)
        N = predictions.shape[0]
        ce_loss = -np.sum(targets*np.log(predictions))/N
        return ce_loss


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

    print("train_X shape: {}".format(train_X.shape))
    print("train_y shape: {}".format(train_y.shape))

    print(f'There are {train_X.shape[0]} observations with {n_feats} features classified into {n_classes} classes.')

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
    clf = LR(fit_intercept=False, penalty='none')
    clf.fit(train_X, train_y)
    print(clf.score(train_X, train_y))
    print(clf.score(dev_X, dev_y))

    # plot
    plot(epochs, train_accs, valid_accs)
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss)


if __name__ == '__main__':
    main()

# check if this works in github