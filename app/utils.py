# app/utils.py

import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from huggingface_hub import hf_hub_download
import streamlit as st

@st.cache_resource
def load_model():
    """
    Loads the trained model from Hugging Face Hub.
    """
    try:
        model_path = hf_hub_download(
            repo_id="AkashSBalsaraf/ExponentAI-Model",
            filename="exponent_recognition_model.h5"
        )
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"⚠️ Could not load model: {e}")
        st.stop()


def preprocess_image(image):
    """
    Preprocess the drawn image for model prediction.
    - Converts to grayscale
    - Inverts colors
    - Resizes to 28x28
    - Normalizes pixel values
    """
    image = image.convert("L")  # Grayscale
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array


def predict_digit(model, image):
    """
    Runs inference on the preprocessed image using the trained model.
    Returns predicted class and confidence.
    """
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    pred_class = np.argmax(predictions)
    confidence = np.max(predictions)
    return pred_class, confidence
