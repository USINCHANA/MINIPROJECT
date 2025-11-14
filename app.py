import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load model
model = load_model('pneumonia_model.keras')

st.title("Pneumonia Detection")
st.write("Upload a chest X-ray image to check for Pneumonia.")

uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg","png","jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('L')
    img = img.resize((224,224))
    img_array = np.expand_dims(np.array(img)/255.0, axis=0)
    
    prediction = model.predict(img_array)[0][0]
    
    if prediction > 0.5:
        st.error(f"Pneumonia Detected! Confidence: {prediction:.2f}")
    else:
        st.success(f"Normal Lung! Confidence: {1-prediction:.2f}")
    
    st.image(img, caption='Uploaded Image', use_column_width=True)
