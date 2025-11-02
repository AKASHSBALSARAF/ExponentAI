import numpy as np
from PIL import Image, ImageOps

def preprocess_image(image):
    """
    Converts an RGBA image from the drawing canvas into the model's required input format.
    Output shape: (1, 28, 28, 1)
    """
    try:
        # Convert RGBA → grayscale
        image = image.convert("L")

        # Invert colors so drawn digit is dark on light
        image = ImageOps.invert(image)

        # Resize to match training shape
        image = image.resize((28, 28))

        # Convert to numpy array and normalize
        img_array = np.array(image).astype("float32") / 255.0

        # Reshape to model input shape (1, 28, 28, 1)
        img_array = np.expand_dims(img_array, axis=(0, -1))

        return img_array

    except Exception as e:
        print(f"[Error] Image preprocessing failed: {e}")
        return None
