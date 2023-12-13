import numpy as np
import matplotlib.pyplot as plt

hidden_size = 2
n_features = 3
n_classes = 1


class MLP(object):
    def __init__(self, A, B, D, K):

        # ensure that the rules of the game are adhered to
        assert -D < A <= B < D, "A, B, D must satisfy -D < A <= B < D"

        self.W1 = np.ones((K, D))
        self.b1 = np.ones((K, 1))

        self.W2 = np.ones((1, K))
        self.b2 = 1
    
    def sign(self, Z):
        return np.where(Z >= 0, 1, -1)
    
    def forward(self, X):
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

    def function_f(self, A, B, x):
        if np.sum(x) in range(A, B):
            return 1
        else:
            return -1

def main():
    A = -1
    B = 3
    D = 5
    K = 2

    mlp = MLP(A, B, D, K)
    # let the input vector be any random vector of size Dx1 with entries -1 or 1
    x = np.random.choice([-1, 1], size=(D, 1))

    output = mlp.forward(x)
    print(" ")
    print("The output of the MLP is: ", output)

    print("The output of the original function would be: ", mlp.function_f(A, B, x))

    
if __name__ == '__main__':
    main()