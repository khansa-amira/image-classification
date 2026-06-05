import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown
import os

@st.cache_resource
def load_model_from_drive(file_id):
    output_model_path = 'model_hewan.keras'
    if not os.path.exists(output_model_path):
        with st.spinner('Sedang mengunduh model, mohon tunggu...'):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, output_model_path, quiet=False, fuzzy=True)
    return tf.keras.models.load_model(output_model_path)

GOOGLE_DRIVE_FILE_ID = '1mZO-sDMrG4muobSGu1EknZBFI7uZCr-O'

model = load_model_from_drive(GOOGLE_DRIVE_FILE_ID)
class_names = ['kucing', 'ikan koi']

st.title("Klasifikasi Gambar Hewan")
st.write("Upload gambar kucing atau ikan koi")

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diupload", use_column_width=True)
    
    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    st.success(f"Hasil: **{predicted_class}**")
    st.info(f"Keyakinan: {confidence:.2f}%")
