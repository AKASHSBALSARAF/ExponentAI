import streamlit as st
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas
from utils import load_model, preprocess

st.set_page_config(page_title="Exponent AI", layout="wide")

st.title("✍️ Exponent AI – Handwritten Exponent Recognition")

# Load model from Hugging Face
model = load_model()

st.markdown("Draw an **expression (like 3⁷)** below 👇")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=12,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key=f"canvas_{st.session_state.get('canvas_key', 0)}"
)

col1, col2 = st.columns(2)

if canvas_result.image_data is not None:
    img = Image.fromarray((canvas_result.image_data[:, :, 0:3]).astype("uint8"))
else:
    img = None

with col1:
    if st.button("🔍 Predict"):
        if img:
            x = preprocess(img)
            if x is not None:
                preds = model.predict(x)
                # Since your model has two outputs
                base_pred, exp_pred = preds
                base_class = np.argmax(base_pred)
                exp_class = np.argmax(exp_pred)
                st.success(f"**Predicted Base Digit:** {base_class}\n\n**Predicted Exponent Digit:** {exp_class}")
            else:
                st.warning("⚠️ Image preprocessing failed.")
        else:
            st.warning("Please draw something first!")

with col2:
    if st.button("🧹 Clear Canvas"):
        st.session_state["canvas_key"] = st.session_state.get("canvas_key", 0) + 1
        st.rerun()

st.markdown("---")
st.caption("Built by Akash S. Balsaraf 🚀")
