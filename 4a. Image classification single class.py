import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # type: ignore

# -------------------------
# 1) Load Fashion-MNIST dataset
# -------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test  = x_test / 255.0

# Add channel dimension
x_train = x_train[..., np.newaxis]
x_test  = x_test[..., np.newaxis]

# Class labels
class_names = [
    "T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# -------------------------
# 2) Build CNN model
# -------------------------
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])

# -------------------------
# 3) Compile model
# -------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -------------------------
# 4) Train model
# -------------------------
print("Training started...")
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# -------------------------
# 5) Function to predict image from path
# -------------------------
def predict_image():
    image_path = input("\nEnter image path: ").strip()

    try:
        img = Image.open(image_path).convert("L").resize((28,28))
    except Exception as e:
        print("Error loading image:", e)
        return

    img = np.array(img) / 255.0

    # Uncomment if prediction is wrong due to background
    # img = 1 - img

    img = img.reshape(1,28,28,1)

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    plt.imshow(img.reshape(28,28), cmap="gray")
    plt.title(f"Predicted: {class_names[predicted_class]}")
    plt.axis("off")
    plt.show()

    print("Predicted Class:", class_names[predicted_class])

# -------------------------
# 6) User input image
# -------------------------
predict_image()


