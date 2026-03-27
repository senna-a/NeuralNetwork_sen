import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize images
x_train = x_train / 255.0
x_test = x_test / 255.0

# Add channel dimension
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# Build CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model
print("Training started...")
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Evaluate model
loss, accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", accuracy * 100, "%")

# Predict a handwritten digit image
def predict_digit(image_path):
    img = Image.open(image_path).convert("L").resize((28, 28))
    img = np.array(img) / 255.0

    # Uncomment if digit is black on white
    # img = 1 - img

    img = img.reshape(1, 28, 28, 1)

    prediction = model.predict(img)
    digit = np.argmax(prediction)

    print("Predicted Digit:", digit)

    plt.imshow(img.reshape(28, 28), cmap="gray")
    plt.axis("off")
    plt.show()

# Input your image
predict_digit("5.png")
