import gradio as gr
import tensorflow as tf
import numpy as np

from PIL import Image


# =========================
# LOAD MODEL
# =========================

model = tf.keras.models.load_model(
    "pneumonia_cnn.keras",
    compile=False
)


# =========================
# PREDICTION FUNCTION
# =========================

def predict_image(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize image
    image = image.resize((224,224))

    # Convert to numpy array
    img_array = np.array(image)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)

    probability = prediction[0][0]

    # Binary classification
    if probability > 0.5:
        result = "PNEUMONIA"
        confidence = probability
    else:
        result = "NORMAL"
        confidence = 1 - probability

    return f"{result} ({confidence:.2%} confidence)"


# =========================
# CREATE INTERFACE
# =========================

interface = gr.Interface(

    fn=predict_image,

    inputs=gr.Image(type="pil"),

    outputs="text",

    title="Pneumonia Detection CNN",

    description="Upload a chest X-ray image."
)


# =========================
# LAUNCH APP
# =========================

interface.launch()