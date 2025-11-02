import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import requests
from io import BytesIO
import streamlit as st

@st.cache_resource
def load_model():
    url = "https://huggingface.co/AkashSBalsaraf/ExponentAI-Model/resolve/main/exponent_recognition_model.h5"
    response = requests.get(url)
    response.raise_for_status()
    model = tf.keras.models.load_model(BytesIO(response.content))
    return model

def preprocess(image_data):
    img = Image.open(image_data).convert("L")
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = 255 - img_array
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array
