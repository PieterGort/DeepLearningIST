import numpy as np

# Defining the MLP class with necessary methods
class MLP(object):
    def __init__(self, W1, b1, W2, b2):
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
    
    def sign(self, Z):
        return np.where(Z >= 0, 1, -1)
    
    def Relu(self, Z):
        return np.maximum(0, Z)
    
    def forward(self, X):
        Z1 = self.W1.dot(X) + self.b1
        h1 = self.sign(Z1)
        Z2 = self.W2.dot(h1) + self.b2
        h2 = self.sign(Z2)
        return h2
    
    def function_f(self, A, B, x):
        return 1 if np.sum(x) in range(A, B+1) else -1

# Defining the main routine that searches for the correct weights and biases
def find_weights_and_biases():
    A = 0
    B = 1
    D = 2
    K = 2
    
    # Defining input vectors
    inputs = [np.array([[1], [1]]), np.array([[1], [-1]]), 
              np.array([[-1], [1]]), np.array([[-1], [-1]])]
    
    # Iterate over combinations of integer weights and biases
    for w1_11 in range(-D, D+1):
        for w1_12 in range(-D, D+1):
            for w1_21 in range(-D, D+1):
                for w1_22 in range(-D, D+1):
                    for b1_1 in range(-D, D+1):
                        for b1_2 in range(-D, D+1):
                            for w2_1 in range(-D, D+1):
                                for w2_2 in range(-D, D+1):
                                    for b2 in range(-D, D+1):
                                        W1 = np.array([[w1_11, w1_12], [w1_21, w1_22]])
                                        b1 = np.array([[b1_1], [b1_2]])
                                        W2 = np.array([[w2_1, w2_2]])
                                        b2 = np.array([b2])
                                        
                                        # Initialize MLP with current weights and biases
                                        mlp = MLP(W1, b1, W2, b2)
                                        
                                        # Check if the outputs match for all input vectors
                                        match = True
                                        for x in inputs:
                                            output = mlp.forward(x)
                                            expected_output = mlp.function_f(A, B, x)
                                            if output != expected_output:
                                                match = False
                                                break
                                        
                                        # If a matching configuration is found, return the weights and biases
                                        if match:
                                            return W1, b1, W2, b2
    return None  # If no configuration is found

# Run the search routine
W1, b1, W2, b2 = find_weights_and_biases()

print("Weights W1: ", W1)
print("")
print("Bias b1: ", b1)
print("")
print("Weights W2: ", W2)
print("")
print("Bias b2: ", b2)

