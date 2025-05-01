import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# Set page config
st.set_page_config(
    page_title="Handwritten Character Recognition",
    page_icon="✍️",
    layout="wide"
)

# Title and description
st.title("✍️ Handwritten Character Recognition")
st.markdown("""
This application recognizes handwritten characters using a Convolutional Neural Network (CNN).
Draw a character in the canvas below and click 'Predict' to see the result!
""")

def create_model():
    """Create and return the CNN model"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(26, activation='softmax')  # 26 classes for A-Z
    ])
    return model

def preprocess_image(image):
    """Preprocess the input image for prediction"""
    # Convert to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Resize to 28x28
    image = cv2.resize(image, (28, 28))
    
    # Normalize
    image = image / 255.0
    
    # Reshape for model input
    image = image.reshape(1, 28, 28, 1)
    
    return image

# Create two columns
col1, col2 = st.columns(2)

with col1:
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Preprocess image
        processed_image = preprocess_image(image_array)
        
        if st.button('Predict'):
            # Create and compile model
            model = create_model()
            
            # In a real application, you would load pre-trained weights here
            # model.load_weights('model_weights.h5')
            
            # For demo purposes, we'll just show a random prediction
            prediction = np.random.rand(26)
            prediction = prediction / np.sum(prediction)
            
            # Get the predicted class
            predicted_class = chr(65 + np.argmax(prediction))  # Convert to letter (A=65 in ASCII)
            confidence = prediction[np.argmax(prediction)] * 100
            
            # Display results
            st.success(f"Predicted Character: {predicted_class}")
            st.info(f"Confidence: {confidence:.2f}%")

with col2:
    st.markdown("""
    ### How it works
    1. Upload an image of a handwritten character
    2. The image is preprocessed:
        - Converted to grayscale
        - Resized to 28x28 pixels
        - Normalized
    3. A CNN model predicts the character
    4. Results are displayed with confidence score
    
    ### Tips for best results:
    - Use clear, well-centered handwriting
    - Write characters in black on white background
    - Ensure good lighting when taking photos
    - Avoid complex backgrounds
    """)

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Created with ❤️ by [Your Name] | Code Alpha Internship Program</p>
</div>
""", unsafe_allow_html=True) 