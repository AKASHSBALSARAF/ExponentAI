import streamlit as st
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import gzip
import io

# =====================================
# Load Keras Model (.h5.gz)
# =====================================
@st.cache_resource
def load_model():
    try:
        with gzip.open("model/exponent_recognition_model.h5.gz", "rb") as f:
            model_bytes = f.read()
        model = tf.keras.models.load_model(io.BytesIO(model_bytes))
        return model
    except Exception as e:
        st.error(f"⚠️ Model could not be loaded: {e}")
        st.stop()

model = load_model()

# =====================================
# Prediction function
# =====================================
def predict_digit(image):
    # Convert RGBA → grayscale
    img = image.convert("L")

    # Invert dynamically if needed
    np_img = np.array(img)
    if np.mean(np_img) > 127:
        img = ImageOps.invert(img)

    # Resize & normalize
    img = img.resize((28, 28))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    try:
        preds = model.predict(img_array, verbose=0)
        pred_class = int(np.argmax(preds))
        confidence = float(np.max(preds))
        return pred_class, confidence
    except Exception:
        return None, None

# =====================================
# Streamlit UI
# =====================================
st.set_page_config(page_title="Exponent Recognition", layout="centered")
st.title("🧮 Exponent Recognition (Keras Model)")
st.markdown("Draw an exponent (like 2, 3, x²) inside the box below and click **Predict**.")

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=8,
    stroke_color="black",
    background_color="white",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Predict"):
        if canvas_result.image_data is not None:
            image = Image.fromarray(
                (canvas_result.image_data[:, :, 0:3]).astype("uint8"), "RGB"
            )
            pred_class, confidence = predict_digit(image)

            if pred_class is not None and confidence > 0.5:
                st.success(f"Prediction: **{pred_class}**  \nConfidence: `{confidence:.2f}`")
            else:
                st.warning("Try again — cannot recognize input.")
        else:
            st.warning("Please draw something first.")

with col2:
    if st.button("Clear Canvas"):
        st.rerun()  # ✅ modern replacement for experimental_rerun()

st.markdown("---")
st.caption("Built with Streamlit • TensorFlow • Handwritten Exponent Model")
