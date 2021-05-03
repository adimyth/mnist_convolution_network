import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from config import configs
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPool2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from utils import plot_confusion_matrix, plot_history, set_seeds

# set seeds
set_seeds(configs.RANDOM_STATE)

# loading data
df = pd.read_csv(configs.TRAIN_PATH)
X = df.iloc[:, 1:].values
y = df["label"]
print(f"Original Training Data: {df.shape}")
print(f"Label Distribution:\n{df['label'].value_counts(sort=False)}")

# split into train & validation folds
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, stratify=y, random_state=configs.RANDOM_STATE, test_size=0.1
)
print(f"Training Data: {X_train.shape}, {y_train.shape}")
print(f"Validation Data: {X_valid.shape}, {y_valid.shape}")

# normalising data such that value ranges between 0-1
X_train = X_train / 255.0
X_valid = X_valid / 255.0

# reshaping 784 pixels into 28x28 matrix
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_valid = X_valid.reshape(X_valid.shape[0], 28, 28, 1)

# converting labels to one-hot encode representation
y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)

# CNN Model - Read more info in README
input_shape = (28, 28, 1)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.20))
model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(configs.NUM_CLASSES, activation="softmax"))

# defining optimizer, loss function & metric
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# training the model
history = model.fit(
    X_train,
    y_train,
    batch_size=configs.BATCH_SIZE,
    validation_data=(X_valid, y_valid),
    epochs=configs.EPOCHS,
)

# save the model
model.save("../models/mnist_classifier.h5")

# plot training history
plot_history(history)

# print final training & validation accuracy
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
print(f"\nFinal Training Loss: {train_loss}\nFinal Training Accuracy: {train_acc}")
valid_loss, valid_acc = model.evaluate(X_valid, y_valid, verbose=0)
print(f"\nFinal Validation Loss: {valid_loss}\nFinal Validation Accuracy: {valid_acc}")

# plot confusion matrix - allows classwise comparison
y_pred = model.predict(X_valid)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_valid, axis=1)
# compute the confusion matrix
conf_mtrx = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(conf_mtrx, classes=range(10))
