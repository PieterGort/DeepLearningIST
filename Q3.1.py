import numpy as np
import matplotlib.pyplot as plt

hidden_size = 2
n_features = 3
n_classes = 1


class MLP(object):
    def __init__(self, A, B, D, K):

        # ensure that the rules of the game are adhered to
        assert -D < A <= B < D, "A, B, D must satisfy -D < A <= B < D"

        self.W1 = np.array([[1, 1], [1, 1]])
        self.b1 = np.array([[1], [1]])

        self.W2 = np.array([[1, 1]])
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
        losses = []
        for x_i, y_i in zip(X, y):
            # Forward propagation
            z1 = np.dot(x_i, self.W1.T) + self.b1
            h1 = self.ReLu(z1)
            z2 = np.dot(h1, self.W2.T) + self.b2
            h2 = self.softmax_MLP(z2)

            # Backward propagation
            y_one_hot = np.zeros(self.W2.shape[0])
            y_one_hot[y_i] = 1

            loss = self.cross_entropy(h2, y_one_hot)
            losses.append(loss)

            dZ2 = h2 - y_one_hot
            dW2 = np.outer(dZ2, h1)
            db2 = dZ2

            dh1 = np.dot(dZ2, self.W2)
            dZ1 = dh1 * self.relu_derivative(z1)
            dW1 = np.outer(dZ1, x_i)
            db1 = dZ1

            # Update weights and biases
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1

        return np.mean(losses)
        
    def cross_entropy(self, prediction, target, epsilon=1e-12):
        prediction = np.clip(prediction, epsilon, 1. - epsilon)
        celoss = -np.sum(target*np.log(prediction+1e-9))
        return celoss

    def function_f(self, A, B, x):
        if np.sum(x) in range(A, B):
            return 1
        else:
            return -1

def main():
    A = -1
    B = 1
    D = 2
    K = 2

    mlp = MLP(A, B, D, K)
    # let the input vector be any random vector of size Dx1 with entries -1 or 1
    # make the vector [-1, 1]^T
    x1 = np.array([[1], [1]])
    x2 = np.array([[1], [-1]])
    x3 = np.array([[-1], [1]])
    x4 = np.array([[-1], [-1]])


    output1 = mlp.forward(x1)
    output2 = mlp.forward(x2)
    output3 = mlp.forward(x3)
    output4 = mlp.forward(x4)
    print(" ")
    print("Output for x1: ", output1)
    print("Output for x2: ", output2)
    print("Output for x3: ", output3)
    print("Output for x4: ", output4)
    print(" ")
    print("Actual output for x1: ", mlp.function_f(A, B, x1))
    print("Actual output for x2: ", mlp.function_f(A, B, x2))
    print("Actual output for x3: ", mlp.function_f(A, B, x3))
    print("Actual output for x4: ", mlp.function_f(A, B, x4))

    print("Weights W1: ", mlp.W1)
    print("")
    print("Bias b1: ", mlp.b1)
    print("")
    print("Weights W2: ", mlp.W2)
    print("")
    print("Bias b2: ", mlp.b2)

    
if __name__ == '__main__':
    main()



