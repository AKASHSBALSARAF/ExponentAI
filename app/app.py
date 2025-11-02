import streamlit as st
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps

# =====================================
# Load TFLite model (cached for speed)
# =====================================
@st.cache_resource
def load_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="model/exponent_recognition_model.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error("Model could not be loaded. Please verify the file path and name.")
        st.stop()

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =====================================
# Prediction function
# =====================================
def predict_digit(image):
    # Convert RGBA to grayscale
    img = image.convert("L")

    # Invert only if background is dark
    np_img = np.array(img)
    if np.mean(np_img) > 127:
        img = ImageOps.invert(img)

    # Resize to 28x28 and normalize
    img = img.resize((28, 28))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    try:
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        pred_class = int(np.argmax(output_data))
        confidence = float(np.max(output_data))
        return pred_class, confidence
    except Exception:
        return None, None

# =====================================
# Streamlit UI
# =====================================
st.set_page_config(page_title="Exponent Recognition", layout="centered")
st.title("Exponent Recognition")
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

            if pred_class is not None and confidence > 0.6:
                st.subheader(f"Prediction: {pred_class}")
                st.write(f"Confidence: {confidence:.2f}")
            else:
                st.warning("Try again — cannot recognize input.")
        else:
            st.warning("Please draw something first.")

with col2:
    if st.button("Clear Canvas"):
        st.rerun()

st.markdown("---")
st.caption("Built with TensorFlow Lite • Streamlit • Handwritten Exponent Dataset")
