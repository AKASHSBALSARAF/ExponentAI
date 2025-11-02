import streamlit as st
from PIL import Image, ImageOps, ImageDraw
import numpy as np
from .utils import load_model, preprocess
import streamlit_drawable_canvas as st_canvas

st.set_page_config(page_title="Exponent AI", layout="wide")

st.title("✍️ Exponent AI – Handwritten Exponent Recognition")

# Load model from Hugging Face
model = load_model()

st.markdown("Draw an **exponent (power)** below 👇")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=12,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

col1, col2 = st.columns(2)

if canvas_result.image_data is not None:
    img = Image.fromarray((canvas_result.image_data[:, :, 0:3]).astype("uint8"))
else:
    img = None

with col1:
    if st.button("Predict"):
        if img:
            x = preprocess(img)
            preds = model.predict(x)
            pred_class = np.argmax(preds)
            st.success(f"🔢 Predicted Exponent: **{pred_class}**")
        else:
            st.warning("Please draw something first!")

with col2:
    if st.button("Clear Canvas"):
        st.session_state["canvas"] = None
        st.rerun()

st.markdown("---")
st.caption("Built by Akash S. Balsaraf 🚀")
