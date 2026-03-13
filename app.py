import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- 1. Load the trained model ---
# This path should point to where your 'Pretrained_model.keras' is stored
# If running in Colab, you might need to download it or ensure it's accessible.
@st.cache_resource # Cache the model loading to improve performance
def load_my_model():
    model = tf.keras.models.load_model('Pretrained_model.keras')
    return model

model = load_my_model()

# --- 2. Define image preprocessing function ---
# This should match the preprocessing done during training (e.g., EfficientNet preprocess_input)
def preprocess_image(image):
    image = image.resize((380, 380)) # Resize to target size used in training
    image_array = np.array(image) # Convert PIL image to numpy array
    image_array = np.expand_dims(image_array, axis=0) # Add batch dimension

    # Assuming EfficientNet's preprocess_input was used during training
    # Make sure to import it: `from tensorflow.keras.applications.efficientnet import preprocess_input`
    # image_array = preprocess_input(image_array)
    return image_array

# --- 3. Define class labels ---
# These should match the order of your model's output classes
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# --- 4. Streamlit App Layout ---
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor Classification")
st.write("Upload an MRI image to predict if a brain tumor is present and its type.")

# --- 5. Image Upload Widget ---
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # --- 6. Preprocess and Predict ---
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)

    # Get the class with the highest probability
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_index]
    confidence = np.max(predictions) * 100

    # --- 7. Display Results ---
    st.success(f"Prediction: **{predicted_class.upper()}**")
    st.write(f"Confidence: **{confidence:.2f}%**")

    st.subheader("Raw Predictions (Probabilities):")
    # Display probabilities for all classes
    for i, prob in enumerate(predictions[0]):
        st.write(f"- {class_labels[i].capitalize()}: {prob*100:.2f}%")

st.markdown("---")
st.info("Disclaimer: This model is for educational and experimental purposes only and should not be used for medical diagnosis.")
