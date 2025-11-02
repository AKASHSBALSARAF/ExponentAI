import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import requests
from io import BytesIO
from utils import preprocess_image

st.set_page_config(page_title="Exponent AI", page_icon="∑", layout="centered")

# Load model from Hugging Face (cached)
@st.cache_resource
def load_model():
    try:
        MODEL_URL = "https://huggingface.co/AkashSBalsaraf/ExponentAI-Model/resolve/main/exponent_recognition_model.h5"
        response = requests.get(MODEL_URL)
        response.raise_for_status()
        model = tf.keras.models.load_model(BytesIO(response.content))
        return model
    except Exception as e:
        st.error(f"⚠️ Model could not be loaded: {e}")
        st.stop()

model = load_model()

st.title("Exponent AI ✴")
st.markdown("Draw a handwritten exponent and the model will recognize it.")

# Drawing Canvas
from streamlit_drawable_canvas import st_canvas
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=12,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Prediction function
def predict_digit(image):
    img_array = preprocess_image(image)
    if img_array is None:
        return None, 0
    pred = model.predict(img_array)
    pred_class = np.argmax(pred)
    confidence = np.max(pred)
    return pred_class, confidence

# Predict Button
if st.button("Predict"):
    if canvas_result.image_data is not None:
        image = Image.fromarray((canvas_result.image_data).astype("uint8"), "RGBA")
        pred_class, confidence = predict_digit(image)
        if pred_class is not None and confidence > 0.5:
            st.success(f"Predicted Exponent: **{pred_class}** (Confidence: {confidence:.2f})")
        else:
            st.warning("Try again — could not recognize the input.")
    else:
        st.info("Please draw something first.")
