import numpy as np

# Activation functions
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class FeedForward2HL:
    def __init__(self):
        # Given weights
        self.W1 = np.array([0.5, -0.2, 0.3])   # Hidden Layer 1
        self.W2 = np.array([0.4, 0.1, -0.5])   # Hidden Layer 2
        
        # Bias terms
        self.b1 = 0.0
        self.b2 = 0.0
        
        # Output weights (h1 and h2 → output)
        self.Wout = np.array([1.0, 1.0])
        self.bout = 0.0

    def forward(self, x):
        # Hidden Layer 1 (ReLU)
        z1 = np.dot(x, self.W1) + self.b1
        h1 = relu(z1)

        # Hidden Layer 2 (Sigmoid)
        z2 = np.dot(x, self.W2) + self.b2
        h2 = sigmoid(z2)

        # Output Layer (Sigmoid)
        z3 = h1 * self.Wout[0] + h2 * self.Wout[1] + self.bout
        output = sigmoid(z3)

        return output


# Run model
#(free=1, win=0, offer=1)
x = np.array([1, 0, 1])        
model = FeedForward2HL()
result = model.forward(x)

print("Model Output:", result)
