import tensorflow as tf

print("Program started")

# Input and target
X = tf.constant([[1.0, 0.0, 1.0]])
y = tf.constant([[1.0]])

# Initialize weights and biases
W1 = tf.Variable(tf.random.normal((3, 2)))
b1 = tf.Variable(tf.zeros((1, 2)))

W2 = tf.Variable(tf.random.normal((2, 1)))
b2 = tf.Variable(tf.zeros((1, 1)))

learning_rate = 0.1
epochs = 5

# Sigmoid activation function
def sigmoid(x):
    return tf.math.sigmoid(x)

# Forward propagation
def forward_pass(X):
    z1 = tf.matmul(X, W1) + b1
    a1 = sigmoid(z1)

    z2 = tf.matmul(a1, W2) + b2
    y_pred = sigmoid(z2)

    return y_pred

# Training loop
for epoch in range(epochs):

    with tf.GradientTape() as tape:
        y_pred = forward_pass(X)
        loss = tf.reduce_mean(tf.square(y - y_pred))

    # Compute gradients
    gradients = tape.gradient(loss, [W1, b1, W2, b2])

    # Update weights and biases
    W1.assign_sub(learning_rate * gradients[0])
    b1.assign_sub(learning_rate * gradients[1])
    W2.assign_sub(learning_rate * gradients[2])
    b2.assign_sub(learning_rate * gradients[3])

    print(f"Epoch {epoch + 1}, Loss: {loss.numpy()}", flush=True)

print("Final Predicted Output:", y_pred.numpy())
print("Program ended")

