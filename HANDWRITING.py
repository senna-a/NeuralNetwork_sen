import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. LOAD THE MNIST DATA
# -----------------------------
# MNIST = 60,000 training images + 10,000 test images of digits (0–9)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Train:", x_train.shape, y_train.shape)
print("Test :", x_test.shape, y_test.shape)

# -----------------------------
# 2. PREPROCESSING
# -----------------------------
# Pixels are 0–255 → scale to 0–1 for easier learning
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension: (28, 28) → (28, 28, 1)
# CNNs expect (height, width, channels)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# -----------------------------
# 3. BUILD A SIMPLE CNN MODEL
# -----------------------------
model = tf.keras.Sequential([
    # 1st convolution layer: looks for simple patterns (lines, curves, etc.)
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=(28, 28, 1)
    ),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),  # reduces image size

    # Flatten → turn 2D feature maps into 1D vector
    tf.keras.layers.Flatten(),

    # Small hidden layer
    tf.keras.layers.Dense(64, activation="relu"),

    # Output layer: 10 neurons = digits 0–9, softmax gives probabilities
    tf.keras.layers.Dense(10, activation="softmax")
])

model.summary()

# -----------------------------
# 4. COMPILE THE MODEL
# -----------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",  # labels are 0–9 integers
    metrics=["accuracy"]
)

# -----------------------------
# 5. TRAIN THE MODEL
# -----------------------------
# validation_split=0.1 → 10% of training data used to check performance
history = model.fit(
    x_train, y_train,
    epochs=3,          # keep small at first so it trains fast
    batch_size=64,
    validation_split=0.1
)

# -----------------------------
# 6. TEST ON UNSEEN DATA
# -----------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")

# -----------------------------
# 7. PREDICT ONE EXAMPLE
# -----------------------------
idx = np.random.randint(len(x_test))  # pick random test image
img = x_test[idx]
true_label = y_test[idx]

# Model expects batch dimension → (1, 28, 28, 1)
img_batch = np.expand_dims(img, axis=0)

pred = model.predict(img_batch, verbose=0)
predicted_label = np.argmax(pred[0])

print("\nTrue label     :", true_label)
print("Predicted label:", predicted_label)

plt.imshow(img.squeeze(), cmap="gray")
plt.title(f"True: {true_label} | Pred: {predicted_label}")
plt.axis("off")
plt.show()
