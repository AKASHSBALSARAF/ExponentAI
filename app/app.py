import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import requests
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps
import io, os

st.set_page_config(page_title="Exponent AI", page_icon="∑", layout="centered")

st.title("Exponent AI")
st.markdown("### Handwritten Exponent Recognition")
st.caption("Draw an exponent (0–9, x², etc.) below and let the model predict it.")

# Load the model from Hugging Face
@st.cache_resource
def load_model():
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

model = load_model()

# Canvas settings
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

# Prediction function
def predict_digit(image):
    image = image.convert("L")  # grayscale
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    predictions = model.predict(img_array)
    pred_class = np.argmax(predictions)
    confidence = np.max(predictions)
    return pred_class, confidence

# Prediction area
if st.button("Predict"):
    if canvas_result.image_data is not None:
        img = Image.fromarray((canvas_result.image_data[:, :, 0:3]).astype("uint8"))
        pred, conf = predict_digit(img)
        if conf < 0.7:
            st.warning("Try again — cannot recognize input.")
        else:
            st.success(f"Prediction: **{pred}** (Confidence: {conf:.2f})")
    else:
        st.info("Please draw something first!")

# Clear canvas button (simple rerun)
if st.button("Clear Canvas"):
    st.session_state["canvas"] = None
    st.rerun()

st.markdown("---")
st.caption("Created by **Akash S. Balsaraf** | Powered by TensorFlow & Hugging Face")
