import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# -------------------------
# 1) Load Fashion-MNIST dataset
# -------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Select only two classes: 0 (T-shirt) and 1 (Trouser)
train_filter = np.where((y_train == 0) | (y_train == 1))
test_filter  = np.where((y_test == 0) | (y_test == 1))

x_train, y_train = x_train[train_filter], y_train[train_filter]
x_test, y_test   = x_test[test_filter], y_test[test_filter]

# Normalize images
x_train = x_train / 255.0
x_test  = x_test / 255.0

# Add channel dimension
x_train = x_train[..., np.newaxis]
x_test  = x_test[..., np.newaxis]

# -------------------------
# 2) Build CNN model (Binary)
# -------------------------
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dense(1, activation="sigmoid")
])

# -------------------------
# 3) Compile model
# -------------------------
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -------------------------
# 4) Train model
# -------------------------
print("Training started...")
model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

# -------------------------
# 5) Evaluate model
# -------------------------
loss, accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", accuracy * 100, "%")

# -------------------------
# 6) Predict and display sample image
# -------------------------
prediction = model.predict(x_test[:1])
predicted_label = int(prediction[0] > 0.5)

class_names = ["T-shirt/Top", "Trouser"]

plt.imshow(x_test[0].reshape(28,28), cmap="gray")
plt.title(f"Predicted: {class_names[predicted_label]}")
plt.axis("off")
plt.show()
