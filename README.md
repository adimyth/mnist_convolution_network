# TensorFlow Serving
![Tensorflow](resources/tensorflow.png)
Production ready model serving
* Part of TF Extended (TFX) Ecosystem
* Internally used at Google
* Highly scalable model serving solution
* Works well for models upto 2GB (sufficient for most cases)

## Why Flask is Ineffecient
* No consistent APIs
* No consistent payloads
* No model versioning
* No mini-batching support
* Ineffecient for large models

## Saving the Model
Model gets converted to protobuf file structure. Saves model variables, graph & graph's metadata

```python
import tensorflow as tf

tf.saved_model.save(
model,
export_dir="/tmp/saved_model",
signatures=None
)
```
This would create a folder structure as below. Everything will be timestamped.
```bash
saved_model
├── assets
├── saved_model.pb
└── variables
    ├── variables.data-00000-of-00001
    └── variables.index
```
However, this doesn't just suffice as TF Serving complains about no version being specified. You will have to create a folder `1` inside `saved_model` & move the contents of that directory inside it.

## TensorFlow Serving on MacOS
TF-Serving is not available for Windows or macOS. So, the only option is to use Docker
### Pulling Server Image
```bash
docker pull tensorflow/serving
```

### Running a serving image
The serving images (both CPU and GPU) have the following properties:

* Port 8500 exposed for gRPC
* Port 8501 exposed for the REST API
* `MODEL_NAME` env variable set to "digit_recognizer". Defaults to "model"

### Serving with Docker
Inside the root directory.
```bash
docker run -d --name tfserving -p 8501:8501 \
  -v "$PWD/saved_model:/models/digit_recognizer" \
  -e MODEL_NAME=digit_recognizer -t tensorflow/serving
```
* `-d` - Runs in daemon mode
* `-p 8501:8501` - Maps port 8501 on host to container
* `--name tfserving`  - Name of the container
* `--mount` - Bind mounts the local folder on to the container. Allows accessing the saved model from within the container
* `-e` - Sets environment variable. Sets `MODEL_NAME` to "digit_recognizer". This will form the part of URL endpoint
* `-t` - Docker image to use (tensorflow/serving here)

## Application
Tensorflow serving provides two endpoints - REST and gRPC.
### REST
* Standard HTTP Post requests
* Response is a JSON Body with the prediction
* Request from the default or specific model

Default URL structure - `http://localhost:8501/v1/models/{MODEL_NAME}`

A sample script which reads data from [Streamlit Drawable Canvas](https://github.com/andfanilo/streamlit-drawable-canvas) and sends it as a HTTP Post request to the deployed model -

```python
import streamlit as st

# Code to add drawable canvas. Refer link above
canvas_result = ...

# Read image & resize it
img = cv2.resize(canvas_result.image_data.astype("uint8"), (28, 28))
# Convert to grayscale
test_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Add an additional dimension
test_img = test_img.reshape(1, 28, 28, 1)

# Prepare headers & data to be sent in the POST request
json_data = json.dumps({"instances": test_img.tolist()})
headers = {"content-type": "application/json"}
# Send the request to the Prediction API
response = requests.post(endpoint, data=json_data, headers=headers)
prediction = tf.argmax(response.json()["predictions"][0])
print(f"Prediction: {prediction}")
```