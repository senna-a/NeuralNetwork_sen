import tensorflow as tf
import pandas as pd
import numpy as np

# 1. LOAD THE CSV DATASET
df = pd.read_excel("spam_dataset_25rows.xlsx")  # uploaded file

emails = df["email"].astype(str).tolist()
labels = df["label"].astype("float32").values

# 2. CREATE A TENSORFLOW DATASET

batch_size = 2
ds = tf.data.Dataset.from_tensor_slices((emails, labels)).batch(batch_size)

# 3. TEXT VECTORIZATION
max_tokens = 1000
sequence_length = 20

vectorize = tf.keras.layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=sequence_length
)

# Only emails, no labels
text_only_ds = ds.map(lambda x, y: x)
vectorize.adapt(text_only_ds)

# 4. BUILD MODEL
model = tf.keras.Sequential([
    vectorize,
    tf.keras.layers.Embedding(max_tokens, 16),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(ds, epochs=10, verbose=1)
print("Training accuracy:", history.history["accuracy"][-1])

# 6. TEST ON NEW EMAILS
test_emails = [
    "You have won a free gift card",
    "Let's meet tomorrow to discuss the assignment",
    "free win, click here now"
]

test_array = np.array(test_emails, dtype=object)
predictions = model.predict(test_array)

for email, p in zip(test_emails, predictions):
    print("\nEmail:", email)
    print("Spam Probability:", float(p))
    print("Prediction:", "SPAM" if p >= 0.5 else "NOT SPAM")
