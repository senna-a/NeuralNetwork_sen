import tensorflow as tf
import numpy as np

emails = [
    "Win a free iPhone now",                # spam
    "Congratulations, you won a lottery",   # spam
    "Get cheap loans instantly",            # spam
    "Meeting at 10 am tomorrow",            # not spam
    "Please find the attached report",      # not spam
    "Let's have lunch today"                # not spam
]


labels = np.array([1, 1, 1, 0, 0, 0], dtype="float32")

batch_size = 2
ds = tf.data.Dataset.from_tensor_slices((emails, labels)).batch(batch_size)

max_tokens = 1000          # vocab size
sequence_length = 20       # max words per email

vectorize = tf.keras.layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=sequence_length
)

text_only_ds = ds.map(lambda x, y: x)
vectorize.adapt(text_only_ds)

# 3. neural network model
model = tf.keras.Sequential([
    vectorize,                                   # converts text -> integer sequence
    tf.keras.layers.Embedding(max_tokens, 16),   # word embeddings
    tf.keras.layers.GlobalAveragePooling1D(),    # average over words
    tf.keras.layers.Dense(8, activation="relu"), # hidden layer (like your ReLU neuron)
    tf.keras.layers.Dense(1, activation="sigmoid") # output: spam probability
])

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# 4. Train the model

history = model.fit(ds, epochs=20, verbose=0)

print("Training complete.")
print("Final accuracy:", history.history["accuracy"][-1])

# 5. Test on new emails
test_emails = [
    "You won a free vacation",
    "Can we reschedule our meeting to Monday?",
    "Limited time offer, claim your prize now"
]

# Convert list → NumPy array (or tf.constant)
test_emails_array = np.array(test_emails, dtype=object)
# OR: test_emails_array = tf.constant(test_emails)

predictions = model.predict(test_emails_array)

for email, p in zip(test_emails, predictions):
    print("\nEmail:", email)
    print("Spam probability:", float(p))
    print("Predicted class:", "SPAM" if p >= 0.5 else "NOT SPAM")
