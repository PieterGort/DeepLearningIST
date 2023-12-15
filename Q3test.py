import numpy as np
import matplotlib.pyplot as plt

hidden_size = 2
n_features = 3
n_classes = 1

class MLP(object):
    def __init__(self, A, B, D, K):
        assert -D < A <= B < D, "A, B, D must satisfy -D < A <= B < D"
        self.W1 = np.array([[1, 1], [-1,-1]])
        self.b1 = np.array([[-A], [B]])
        self.W2 = np.array([[1, 1]])
        self.b2 = -2

    def sign(self, Z):
        return np.where(Z >= 0, 1, -1)

    def forward(self, X):
        Z1 = self.W1.dot(X) + self.b1
        h1 = self.sign(Z1)
        Z2 = self.W2.dot(h1) + self.b2
        h2 = self.sign(Z2)
        return h2
    
    def function_f(self, A, B, x):
        if np.sum(x) in range(A, B):
            return 1
        else:
            return -1

def main():
    A = 0
    B = 1
    D = 2
    K = 2

    mlp = MLP(A, B, D, K)
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
