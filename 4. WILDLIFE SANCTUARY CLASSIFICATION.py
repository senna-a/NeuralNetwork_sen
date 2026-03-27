
# IMPORTS
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing import image

# LOAD CIFAR10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# KEEP ONLY BIRD(2), CAT(3), DOG(5)
wanted_classes = [2,3,5]

def filter_classes(x, y):
    mask = np.isin(y, wanted_classes).flatten()
    x = x[mask]
    y = y[mask]

    # remap labels:
    # bird->0, cat->1, dog->2
    mapping = {2:0, 3:1, 5:2}
    y = np.array([mapping[int(label)] for label in y])
    return x, y

x_train, y_train = filter_classes(x_train, y_train)
x_test, y_test = filter_classes(x_test, y_test)

class_names = ["bird","cat","dog"]

# DATA AUGMENTATION (improves accuracy)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# BUILD BETTER CNN
model = models.Sequential([

    data_augmentation,

    layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(64,(3,3),activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(128,(3,3),activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128,activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(3,activation='softmax')
])

# COMPILE
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# TRAIN
model.fit(
    x_train,
    y_train,
    epochs=15,
    validation_data=(x_test,y_test)
)

# SAVE
model.save("animal_classifier.h5")

# TEST WITH YOUR IMAGE
img_path = "animal.jpg"

img = image.load_img(img_path,target_size=(32,32))
img_array = image.img_to_array(img)/255.0
img_array = np.expand_dims(img_array,axis=0)

prediction = model.predict(img_array)

predicted_class = class_names[np.argmax(prediction)]

print("Predicted:", predicted_class)

plt.imshow(img)
plt.title(predicted_class)
plt.axis("off")
plt.show()
