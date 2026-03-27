
#Trains (or loads) a small CNN that recognizes two digits placed side-by-side (28x56 image).
#Prompts you to TYPE two digits (0-9). For each typed digit the script samples an MNIST example.
#shows the combined image and the model's prediction.

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

MODEL_PATH = "two_digit_model.h5"

# -------------------------
# 1) Load MNIST and normalize
# -------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# -------------------------
# 2) Utility: create two-digit images dataset
# -------------------------
def make_two_digit_dataset(images, labels, count):
    """Return images (count,28,56,1) and labels (count,2)."""
    n = len(images)
    imgs = np.empty((count, 28, 56), dtype="float32")
    labs = np.empty((count, 2), dtype="int32")
    for i in range(count):
        i1 = np.random.randint(0, n)
        i2 = np.random.randint(0, n)
        a = images[i1]
        b = images[i2]
        imgs[i] = np.concatenate([a, b], axis=1)   # horizontal concat -> 28x56
        labs[i, 0] = labels[i1]
        labs[i, 1] = labels[i2]
    imgs = np.expand_dims(imgs, -1)  # (count,28,56,1)
    return imgs, labs

# -------------------------
# 3) Build the model
# -------------------------
def build_model():
    inp = tf.keras.Input(shape=(28, 56, 1))
    x = tf.keras.layers.Conv2D(32, 3, activation="relu")(inp)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    out1 = tf.keras.layers.Dense(10, activation="softmax", name="digit1")(x)
    out2 = tf.keras.layers.Dense(10, activation="softmax", name="digit2")(x)

    model = tf.keras.Model(inputs=inp, outputs=[out1, out2])
    model.compile(
        optimizer="adam",
        loss={"digit1": "sparse_categorical_crossentropy", "digit2": "sparse_categorical_crossentropy"},
        metrics={"digit1": "accuracy", "digit2": "accuracy"},
    )
    return model

# -------------------------
# 4) Load or train model (keeps training small for speed)
# -------------------------
if os.path.exists(MODEL_PATH):
    print("Loading saved model:", MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print("No saved model found. Preparing small training dataset and training a model (fast).")
    two_x_train, two_y_train = make_two_digit_dataset(x_train, y_train, count=12000)
    two_x_val,   two_y_val   = make_two_digit_dataset(x_train, y_train, count=2000)
    model = build_model()
    model.summary()
    model.fit(
        two_x_train,
        {"digit1": two_y_train[:, 0], "digit2": two_y_train[:, 1]},
        validation_data=(two_x_val, {"digit1": two_y_val[:, 0], "digit2": two_y_val[:, 1]}),
        epochs=3,
        batch_size=64,
        verbose=1
    )
    model.save(MODEL_PATH)
    print("Model trained and saved to", MODEL_PATH)

# -------------------------
# 5) Utility: sample MNIST image for a typed digit
# -------------------------
def sample_mnist_digit_image(digit, source_images, source_labels):
    """Return one 28x28 MNIST image (normalized) for the requested digit."""
    indices = np.where(source_labels == digit)[0]
    if indices.size == 0:
        raise ValueError(f"Digit {digit} not found in MNIST.")
    idx = np.random.choice(indices)
    return source_images[idx]

# -------------------------
# 6) Prompt: only typing two digits (0-9)
# -------------------------
print("\nType two digits (0-9). The script will sample MNIST images for those digits.")
d1 = input("Enter first digit (0-9): ").strip()
d2 = input("Enter second digit (0-9): ").strip()

# Validate inputs
if not (d1.isdigit() and d2.isdigit()):
    raise ValueError("Please enter numeric digits 0-9.")
d1 = int(d1); d2 = int(d2)
if not (0 <= d1 <= 9 and 0 <= d2 <= 9):
    raise ValueError("Digits must be in the range 0..9.")

# Sample images for those digits from training set
img1 = sample_mnist_digit_image(d1, x_train, y_train)
img2 = sample_mnist_digit_image(d2, x_train, y_train)

# -------------------------
# 7) Combine, predict, and display
# -------------------------
combined = np.concatenate([img1, img2], axis=1)   # (28,56)
input_tensor = np.expand_dims(np.expand_dims(combined, axis=0), -1)  # (1,28,56,1)

pred1, pred2 = model.predict(input_tensor, verbose=0)
p1 = int(np.argmax(pred1[0])); p2 = int(np.argmax(pred2[0]))

plt.figure(figsize=(6, 3))
plt.imshow(combined.squeeze(), cmap="gray")
plt.title(f"Typed: {d1}{d2}  →  Predicted: {p1}{p2}")
plt.axis("off")
plt.show()

print("\nTyped digits (true):", d1, d2)
print("Model predicted    :", p1, p2)
