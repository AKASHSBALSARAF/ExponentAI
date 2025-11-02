import streamlit as st
import numpy as np
from app.utils import load_model, preprocess

st.set_page_config(page_title="ExponentAI", page_icon="⚡")

st.title("⚡ ExponentAI – Handwritten Exponent Recognition")

model = load_model()

uploaded_file = st.file_uploader("Upload your handwritten digit", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img_array = preprocess(uploaded_file)
    prediction = model.predict(img_array)
    st.success(f"Predicted Exponent: {np.argmax(prediction)}")
