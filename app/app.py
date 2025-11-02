import streamlit as st
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import gzip
import io
import os

# =====================================
# Load Keras Model (.h5.gz)
# =====================================

@st.cache_resource
def load_model():
    """Load TensorFlow model from local .gz file or download from GitHub if not found."""
    try:
        MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
        MODEL_PATH = os.path.join(MODEL_DIR, "exponent_recognition_model.h5.gz")
        GITHUB_URL = "https://github.com/AKASHSBALSARAF/ExponentAI/raw/main/model/exponent_recognition_model.h5.gz"


        os.makedirs(MODEL_DIR, exist_ok=True)


        if not os.path.exists(MODEL_PATH):
            st.warning("📥 Model not found locally. Downloading from GitHub...")
            response = requests.get(GITHUB_URL)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            st.success("✅ Model downloaded successfully!")

 
        with gzip.open(MODEL_PATH, "rb") as f_in:
            decompressed = f_in.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_file:
            temp_file.write(decompressed)
            temp_path = temp_file.name

        model = tf.keras.models.load_model(temp_path)

        os.remove(temp_path)  
        return model

    except Exception as e:
        st.error(f"⚠️ Model could not be loaded: {e}")
        st.stop()

# Load model (cached)
model = load_model()
# =====================================
# Prediction function
# =====================================
def predict_digit(image):

    img = image.convert("L")


    np_img = np.array(img)
    if np.mean(np_img) > 127:
        img = ImageOps.invert(img)


    img = img.resize((28, 28))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)


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
        st.rerun() 

st.markdown("---")
st.caption("Built with Streamlit • TensorFlow • Handwritten Exponent Model")
