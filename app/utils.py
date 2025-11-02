import streamlit as st
import tensorflow as tf
import numpy as np
import requests
import io
from PIL import Image

MODEL_URL = "https://huggingface.co/AkashSBalsaraf/ExponentAI-Model/blob/main/exponent_recognition_model.h5"


@st.cache_resource
def load_model():
    try:
        # Download model from Hugging Face
        response = requests.get(MODEL_URL)
        response.raise_for_status()

        # Load model directly from memory
        model_bytes = io.BytesIO(response.content)
        model = tf.keras.models.load_model(model_bytes)
        st.success("✅ Model loaded successfully from Hugging Face!")
        return model
    except Exception as e:
        st.error(f"⚠️ Model could not be loaded: {e}")
        st.stop()


def preprocess(image):
    """
    Preprocesses the canvas image for model prediction.
    """
    image = image.convert("L")            # Grayscale
    image = image.resize((28, 28))        # Match model input
    img_array = np.array(image) / 255.0   # Normalize
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array
