import streamlit as st
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps

# ==============================
# Load the TensorFlow Lite model
# ==============================
@st.cache_resource
def load_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="model/exponent_recognition_model.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error("Model could not be loaded. Please verify the model file.")
        st.stop()

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =========================================
# Preprocessing and prediction function
# =========================================
def predict_digit(image):
    # Convert RGBA to L and resize
    img = image.convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28))

    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0  # normalize
    img_array = img_array.reshape(1, 28, 28, 1)  # match CNN input shape

    try:
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        pred_class = np.argmax(output_data)
        confidence = np.max(output_data)
        return pred_class, confidence
    except Exception as e:
        return None, None

# ===============================
# Streamlit app UI
# ===============================
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
            # Convert canvas result to image
            image = Image.fromarray((canvas_result.image_data[:, :, 0:3]).astype('uint8'), 'RGB')
            pred_class, confidence = predict_digit(image)

            if pred_class is not None and confidence > 0.7:
                st.subheader(f"Prediction: {pred_class}")
                st.write(f"Confidence: {confidence:.2f}")
            else:
                st.warning("Try again — cannot recognize input.")
        else:
            st.warning("Please draw something first.")

with col2:
    if st.button("Clear Canvas"):
        st.experimental_rerun()

# Footer
st.markdown("---")
st.caption("Built with TensorFlow Lite • Streamlit • Handwritten Exponent Dataset")
