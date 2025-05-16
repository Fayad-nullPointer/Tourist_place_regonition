import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Alexandria Landmarks Classifier",
    page_icon="🏛️",
    layout="centered"
)

# Load the model
@st.cache_resource
def load_prediction_model():
    return load_model("model_cv2.h5")

model = load_prediction_model()

# Constants
img_size = (128, 128)
class_names = ['Alexandria Library', 'Qauitbay citadel', 'almorsy abo alabas']

# Description texts
descriptions = {
    'Alexandria Library': 'The Bibliotheca Alexandrina (Alexandria Library) is a major cultural and research center in Egypt, dedicated to knowledge, innovation, and digital transformation. It offers vast historical archives, digital resources, and AI research collaborations. You can leverage its data, manuscripts, and tourism-related studies to develop an AI system that enhances visitor experiences by providing intelligent recommendations, historical insights, and interactive guides for tourists exploring Alexandria.',
    'Qauitbay citadel': 'The Qaitbay Citadel is a historic fortress in Alexandria, Egypt, built in the 15th century by Sultan Al-Ashraf Qaitbay on the ruins of the ancient Lighthouse of Alexandria. It is a major tourist attraction, offering stunning views of the Mediterranean Sea and rich historical insights.',
    'almorsy abo alabas': 'Al-Mursi Abu Al-Abbas Mosque is one of the most famous and beautiful mosques in Alexandria, Egypt. Built in the 13th century and later renovated in the 20th century, it is dedicated to the Andalusian Sufi saint Abu Al-Abbas Al-Mursi. The mosque is known for its grand Islamic architecture, intricate arabesque designs, and spiritual significance, making it a key religious and tourist attraction.'
}

# Prediction function
def predict_image(img_array):
    # Preprocess the image
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = float(np.max(predictions))
    
    return class_names[predicted_class], confidence

# UI Elements
st.title("Alexandria Landmarks Classifier")
st.write("Upload an image of an Alexandria landmark to identify it and get information.")

# Image upload options
st.subheader("Upload Image")
upload_option = st.radio(
    "Choose upload method:",
    ["Upload from Device", "Take Photo with Camera"]
)

if upload_option == "Upload from Device":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
else:  # Camera option
    uploaded_file = st.camera_input("Take a photo of a landmark")

# Process and predict
if uploaded_file is not None:
    try:
        # Process the uploaded image
        img = Image.open(uploaded_file)
        img_array = image.img_to_array(image.load_img(io.BytesIO(uploaded_file.getvalue()), target_size=img_size))
        
        # Add a spinner during prediction
        with st.spinner('Analyzing landmark...'):
            # Make prediction
            predicted_class, confidence = predict_image(img_array)
        
        # Show prediction results
        st.success(f"Prediction: {predicted_class}")
        st.progress(confidence)
        st.write(f"Confidence: {confidence:.2%}")
        
        # Display the image after prediction
        st.subheader("Uploaded Image")
        st.image(img, caption=f'Identified as: {predicted_class}', use_column_width=True)
        
        # Display information about the landmark
        st.subheader(f"About {predicted_class}")
        st.write(descriptions.get(predicted_class, "No information available for this landmark."))
        
        # Additional information section
        st.subheader("Visitor Information")
        if predicted_class == 'Alexandria Library':
            st.write("📍 **Location**: El Shatby, Alexandria")
            st.write("⏰ **Opening Hours**: 10:00 AM - 7:00 PM")
            st.write("💰 **Entry Fee**: 70 EGP")
        elif predicted_class == 'Qauitbay citadel':
            st.write("📍 **Location**: Eastern Harbor, Alexandria")
            st.write("⏰ **Opening Hours**: 9:00 AM - 5:00 PM")
            st.write("💰 **Entry Fee**: 80 EGP")
        elif predicted_class == 'almorsy abo alabas':
            st.write("📍 **Location**: Anfoushi, Alexandria")
            st.write("⏰ **Opening Hours**: Open for prayers")
            st.write("💰 **Entry Fee**: Free")
    
    except Exception as e:
        st.error(f"Error processing image: {e}")

# Footer
st.markdown("---")
st.write("Alexandria Landmarks Classifier - Computer Vision Project")

# Footer
st.markdown("---")
st.write("Alexandria Landmarks Classifier - Computer Vision Project")
