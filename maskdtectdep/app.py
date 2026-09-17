import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

MODEL_PATH = "mask_detector.keras"
IMG_SIZE = (128, 128)


@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    array = np.array(image) / 255.0
    return np.reshape(array, (1, 128, 128, 3))


def main():
    st.title("Face Mask Detection")
    st.write("Upload a photo of a face to check whether a mask is being worn.")

    model = load_model()

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        return

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    input_tensor = preprocess(image)
    prediction = model.predict(input_tensor)
    label = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    if label == 1:
        st.error(f"No mask detected (confidence: {confidence:.2%})")
    else:
        st.success(f"Mask detected (confidence: {confidence:.2%})")


if __name__ == "__main__":
    main()
