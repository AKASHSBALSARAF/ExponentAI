import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from app.utils import load_model, predict_digit

st.set_page_config(page_title="Exponent AI", page_icon="∑", layout="centered")

st.title("Exponent AI")
st.markdown("### Handwritten Exponent Recognition")
st.caption("Draw an exponent (0–9, x², etc.) below and let the model predict it.")

# Load model
model = load_model()

# Drawing canvas
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

# Predict button
if st.button("Predict"):
    if canvas_result.image_data is not None:
        img = Image.fromarray((canvas_result.image_data[:, :, 0:3]).astype("uint8"))
        pred, conf = predict_digit(model, img)
        if conf < 0.7:
            st.warning("Try again — cannot recognize input.")
        else:
            st.success(f"Prediction: **{pred}** (Confidence: {conf:.2f})")
    else:
        st.info("Please draw something first!")

# Clear button
if st.button("Clear Canvas"):
    st.session_state["canvas"] = None
    st.rerun()

st.markdown("---")
st.caption("Created by **Akash S. Balsaraf** | Powered by TensorFlow & Hugging Face")
