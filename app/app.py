import streamlit as st
import numpy as np
import tensorflow.lite as tflite
from PIL import Image, ImageOps
import io

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Handwritten Exponent Recognition",
    page_icon="∧",
    layout="centered",
)

st.title("Handwritten Exponent Recognition")
st.markdown(
    """
Upload or draw a handwritten exponent.  
The model will predict the value or notify if the input cannot be recognized.
"""
)

# ---------------------------
# Loading tensorflowlite
# ---------------------------
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="model/exponent_recognition_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------------------
# Here's the Drawing Canvas
# ---------------------------
from streamlit_drawable_canvas import st_canvas

st.subheader("Draw your exponent below:")
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=12,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=200,
    width=200,
    drawing_mode="freedraw",
    key="canvas",
)

# ---------------------------
# Finally Prediction Function
# ---------------------------
def predict_image(image: Image.Image):
    # Convert to grayscale and resize to match model input
    image = image.convert("L")
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data)
    confidence = np.max(output_data)

    if confidence < 0.6:
        return None
    return prediction

# ---------------------------
# Submiting & Display Results
# ---------------------------
if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Convert canvas to PIL image
        image = Image.fromarray((canvas_result.image_data[:, :, :3] * 255).astype(np.uint8))
        pred = predict_image(image)
        if pred is not None:
            st.success(f"Predicted Exponent: {pred}")
        else:
            st.warning("Try again — input not recognized.")
    else:
        st.info("Please draw something to predict.")

# ---------------------------
# footer
# ---------------------------
st.markdown(
    """
---
**Model:** Custom CNN trained for handwritten exponent recognition  
**Developer:** Akash S. Balsaraf
"""
)
