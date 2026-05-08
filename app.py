import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image


# =========================
# LOAD MODEL
# =========================

model = tf.keras.models.load_model(
    "pneumonia_model.keras",compile=False
)


# =========================
# PAGE TITLE
# =========================

st.title("Pneumonia Detection CNN")

st.write(
    "Upload a chest X-ray image for prediction."
)


# =========================
# FILE UPLOADER
# =========================

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


# =========================
# PREDICTION
# =========================

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Display image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Convert to RGB
    image = image.convert("RGB")

    # Resize image
    image = image.resize((224,224))

    # Convert image to array
    img_array = np.array(image)

    # Normalize image
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)

    # Get probability
    probability = prediction[0][0]

    # Binary classification
    if probability > 0.5:
        result = "PNEUMONIA"
        confidence = probability
    else:
        result = "NORMAL"
        confidence = 1 - probability

    # Show result
    st.subheader(f"Prediction: {result}")

    st.write(
        f"Confidence: {confidence:.2%}"
    )