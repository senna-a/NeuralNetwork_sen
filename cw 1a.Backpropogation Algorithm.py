import numpy as np

print("Program started")

# Input and expected output
X = np.array([[1, 0, 1]])
y = np.array([[1]])

# Fix random seed
np.random.seed(1)

# Initialize weights and biases
W1 = np.random.rand(3, 2)
b1 = np.random.rand(1, 2)

W2 = np.random.rand(2, 1)
b2 = np.random.rand(1, 1)

learning_rate = 0.1
epochs = 5

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Training loop
for epoch in range(epochs):

    # Forward propagation
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    y_pred = sigmoid(z2)

    # Loss (Mean Squared Error)
    loss = np.mean((y - y_pred) ** 2)

    # Backpropagation
    output_error = y - y_pred
    output_delta = output_error * sigmoid_derivative(y_pred)

    hidden_error = np.dot(output_delta, W2.T)
    hidden_delta = hidden_error * sigmoid_derivative(a1)

    # Update weights and biases
    W2 += np.dot(a1.T, output_delta) * learning_rate
    b2 += output_delta * learning_rate

    W1 += np.dot(X.T, hidden_delta) * learning_rate
    b1 += hidden_delta * learning_rate

    print(f"Epoch {epoch + 1}, Loss: {loss}", flush=True)

print("Final Predicted Output:", y_pred)

