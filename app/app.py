import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import io

from streamlit_drawable_canvas import st_canvas

# -------------------------------------------------------
# Load model
# -------------------------------------------------------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model/exponent_recognition_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# -------------------------------------------------------
# Prediction function
# -------------------------------------------------------
def predict_digit(image):
    image = image.convert("L")
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])
    pred_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return pred_class, confidence

# -------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------
st.set_page_config(page_title="Exponent Recognition", layout="centered")

st.title("Exponent Recognition")
st.markdown("Draw an exponent or digit below. The model will recognize it.")
st.divider()

# Drawing canvas
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=8,
    stroke_color="black",
    background_color="white",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:
    image = Image.fromarray((canvas_result.image_data[:, :, :3]).astype(np.uint8))
    st.image(image, caption="Your Drawing", use_container_width=True)

    if st.button("Predict"):
        pred_class, confidence = predict_digit(image)
        if confidence > 0.75:
            st.success(f"Recognized as: **{pred_class}** (Confidence: {confidence:.2f})")
        else:
            st.warning("Try again — input unclear or not recognized.")
else:
    st.info("Draw something above to start.")

st.divider()
st.caption("Built by Akash S. Balsaraf — ExponentAI (2025)")
