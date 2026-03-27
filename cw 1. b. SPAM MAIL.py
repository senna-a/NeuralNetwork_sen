import math

# Input vector
x = [1, 0, 1]

# Hidden layer weights
h1 = [0.9, -0.2, 0.8]   # ReLU neuron
h2 = [0.1, 0.1, 0.9]    # Sigmoid neuron

# Output layer weights (from h1 and h2)
out = [0.7, 0.2]

# Activation functions
relu = lambda z: max(0, z)
sigmoid = lambda z: 1 / (1 + math.exp(-z))

# Hidden Layer Computations
# Hidden neuron 1 → ReLU
h1_raw = x[0]*h1[0] + x[1]*h1[1] + x[2]*h1[2]
h1_out = relu(h1_raw)

# Hidden neuron 2 → Sigmoid
h2_raw = x[0]*h2[0] + x[1]*h2[1] + x[2]*h2[2]
h2_out = sigmoid(h2_raw)

# Output Layer Computation
# Weighted sum of hidden outputs
z_out = h1_out * out[0] + h2_out * out[1]

# Final output (sigmoid activation)
final_output = sigmoid(z_out)

# Print all values
print("Hidden 1 Raw:", h1_raw)
print("Hidden 1 Output (ReLU):", h1_out)

print("Hidden 2 Raw:", h2_raw)
print("Hidden 2 Output (Sigmoid):", h2_out)

print("Output Raw:", z_out)
print("Final Output (Sigmoid):", final_output)
