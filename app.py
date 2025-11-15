import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load model
model = load_model('pneumonia_model.keras')

st.title("Pneumonia Detection")
st.write("Upload a chest X-ray image to check for Pneumonia.")

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    # Open image and display it
    img = Image.open(uploaded_file).convert('RGB')  # Convert to 3 channels
    st.image(img, caption='Uploaded Image',use_container_width=True)
    
    # Preprocess image for model
    img = img.resize((150,150))                      # Resize to model input size
    img_array = np.array(img)/255.0                  # Normalize
    img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension
    
    # Make prediction
    prediction = model.predict(img_array)[0][0]
    
    # Display result
    if prediction > 0.5:
        st.error(f"Pneumonia Detected! Confidence: {prediction*100:.2f}%")
    else:
        st.success(f"Normal Lung! Confidence: {(1-prediction)*100:.2f}%")
