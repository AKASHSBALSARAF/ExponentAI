import streamlit as st
import tensorflow as tf
import numpy as np
import tempfile
import requests
from PIL import Image


@st.cache_resource
def load_model():
    """
    Downloads the model from Hugging Face and loads it into memory.
    Cached for reuse across sessions.
    """
    try:
        HF_URL = "https://huggingface.co/AkashSBalsaraf/ExponentAI-Model/resolve/main/exponent_recognition_model.h5"

        st.info("🔄 Loading model from Hugging Face... Please wait.")

        # Download the model into a temporary file
        response = requests.get(HF_URL)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        # Load model from actual file path
        model = tf.keras.models.load_model(tmp_path)

        st.success(" Model loaded successfully!")
        return model

    except Exception as e:
        st.error(f" Model could not be loaded: {e}")
        st.stop()


def preprocess(image):
    """
    Preprocess the drawn image for prediction.
    Converts to grayscale, resizes to 28x28, and normalizes.
    """
    try:
        # Convert to grayscale
        image = image.convert("L")

        # Resize to match model input
        image = image.resize((28, 28))

        # Convert to numpy and normalize
        img_array = np.array(image) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        return img_array

    except Exception as e:
        st.error(f"Error while preprocessing image: {e}")
        return None
