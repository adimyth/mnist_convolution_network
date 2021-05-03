import numpy as np
import pandas as pd
from config import configs
from tensorflow.keras.models import load_model

# loading data
df = pd.read_csv(configs.TEST_PATH)
submission = pd.read_csv(configs.SUBMISSION_PATH)
X = df.iloc[:, :].values
print(f"Test Data: {df.shape}")

# normalising data such that value ranges between 0-1
X = X / 255.0

# reshaping 784 pixels into 28x28 matrix
X = X.reshape(X.shape[0], 28, 28, 1)

# load trained cnn
model = load_model("../models/mnist_classifier.h5")

# make prediction
y_pred = model.predict(X)
y_pred = np.argmax(y_pred, axis=1)

# making submission file to be submitted on Kaggle
submission["ImageId"] = list(range(1, df.shape[0] + 1))
submission["Label"] = y_pred
submission.to_csv("submission.csv", index=False)
