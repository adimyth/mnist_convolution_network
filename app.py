import json
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import requests
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf


if __name__ == "__main__":
    endpoint = "http://localhost:8501/v1/models/digit_recognizer:predict"
    st.title("TensorFlow Serving")
    st.image(Image.open("resources/tensorflow.png"))
    st.header("Digit Recognizer")
    SIZE = 250
    st.subheader("Draw any Digit")
    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=SIZE,
        height=SIZE,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        img = cv2.resize(canvas_result.image_data.astype("uint8"), (28, 28))
        rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        st.subheader("Model Input")
        st.image(rescaled)

    if st.button("Predict"):
        test_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        test_img = test_img.reshape(1, 28, 28, 1)

        # Prepare the data that is going to be sent in the POST request
        json_data = json.dumps({"instances": test_img.tolist()})
        headers = {"content-type": "application/json"}
        # Send the request to the Prediction API
        response = requests.post(endpoint, data=json_data, headers=headers)
        prediction = tf.argmax(response.json()["predictions"][0])
        st.success(f"Prediction: {prediction}")
