import streamlit as st
import tensorflow as tf
import numpy as np
import requests
import io
from PIL import Image

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

        # Download model to a temporary file
        response = requests.get(HF_URL)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        model = tf.keras.models.load_model(tmp_path)


def preprocess(image):
    """
    Preprocesses the canvas image for model prediction.
    """
    image = image.convert("L")            # Grayscale
    image = image.resize((28, 28))        # Match model input
    img_array = np.array(image) / 255.0   # Normalize
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array
