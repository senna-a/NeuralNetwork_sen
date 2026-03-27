import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

# 1. Load dataset
data = pd.read_csv("/mnt/data/creditcard.csv")

X = data.drop("Class", axis=1)
y = data["Class"]

# 2. Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 4. Handle class imbalance
weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(weights))

# 5. Build Neural Network
model = Sequential()
model.add(Dense(32, activation="relu", input_dim=X_train.shape[1]))
model.add(Dropout(0.3))
model.add(Dense(16, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(1, activation="sigmoid"))

# 6. Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy", AUC(name="auc")]
)

# 7. Train model
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=256,
    validation_split=0.1,
    class_weight=class_weights,
    verbose=1
)

# 8. Evaluate model
loss, accuracy, auc = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
print("Test AUC:", auc)

# 9. Predictions
y_pred = (model.predict(X_test) > 0.5).astype(int)

# =========================
# MATPLOTLIB VISUALIZATIONS
# =========================

# A. Loss Curve
plt.figure()
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.legend()
plt.show()

# B. AUC Curve
plt.figure()
plt.plot(history.history["auc"], label="Training AUC")
plt.plot(history.history["val_auc"], label="Validation AUC")
plt.xlabel("Epochs")
plt.ylabel("AUC")
plt.title("AUC vs Epochs")
plt.legend()
plt.show()

# C. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.colorbar()

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.show()

