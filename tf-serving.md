# TensorFlow Serving

## Why Flask is Ineffecient
* No consistent APIs
* No consistent payloads
* No model versioning
* No mini-batching support
* Ineffecient for large models

## TF Serving
Production ready model serving
* Part of TF Extended (TFX) Ecosystem
* Internally used at Google
* Highly scalable model serving solution
* Works well for models upto 2GB (sufficient for most cases)

### Exporting TF Model
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
![](https://imgur.com/tjWRa9g.png)

### Model Serving
* Docker images are available for CPU & GPU Hardware. Could be run directly on local machine as well.
* gRPC & REST Endpoints
![serving](https://imgur.com/Dj7oaTU.png)

To run the inference on GPU instead of CPU, just pull the GPU container & change the command to -
```bash
docker run ...
	-t tensorflow/serving:latest-gpu
```

## Running Multiple Models
So you can run multiple models at the same time. For this create a `model_config_list` & pass it as a parameter in the run command.
![multiple_models_config](https://imgur.com/FIJ4B42.png)

![running multiple models](https://imgur.com/k1i5xiq.png)

## Running Multiple Versions
You can use `model_version_policy` inside the `config` (model_config_list) & specify the **timestamps** assosciated with the versions you wanna run. The timestamps can be substitued with readable `keys`

![model-versions](https://imgur.com/Xgv818N.png)

## Inferencing
### REST
* Standard HTTP Post requests
* Response is a JSON Body with the prediction
* Request from the default or specific model

Default URL structure
```
http://{HOST}:{PORT}/v1/models/{MODEL_NAME}
```

Specific model versions
```
http://{HOST}:{PORT}/v1/models/{MODEL_NAME}/versions/{MODEL_VERSION}:predict
```

### gRPC
* Inference data needs to be converted into Protobuf format
* Request types have designated types, e.g. float, int, bytes
* Payloads need to be converted to base64
* Connect to the server via gRPC stubs

### REST vs gRPC
* REST is easy to implement & debug
* RPC is more network effecient, smaller payloads
* RPC can provide much faster inferences

## Meta Information
It is possible to get metadata for the served models. Just call the `/metadata` url.
* Enables to get model metadata on the client side
* Endpoint provides the model signatures (inputs & outputs)

## Batching Inferences
* The server can aggregate inference requsts and compute them as a batch. Basically waits for a limited time before inferencing.
* Effecient use of CPU & GPU hardware
* Useful if models are memory-consuming

Configuring this is super-simple. Just create a text file and specify batch size, timeout, batch threads, etc
```python
max_batch_size { value: 32 }
batch_timeout_macros { value: 1000 }
```

Pass it to the run command above as -
```
docker run
	....
	--batching_parameters_file=batching_parameters.txt
```

## Extreme Optimizations - Nvidia TensorRT

## Observations
1. Because of the standard mechanism of serving the models, the inference code is pretty standard. Just a simple URL post request. This satisfies the first point in **Why Flask is Ineffecient** section