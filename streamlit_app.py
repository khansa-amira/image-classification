import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown
import os

# Fungsi untuk mengunduh dan memuat model dari Google Drive
@st.cache_resource
def load_model_from_drive(file_id):
    # Nama file lokal tempat menyimpan model di server Streamlit
    # Ubah ekstensi ke .h5 jika model Anda berformat .h5
    output_model_path = 'model_hewan.h5' 
    
    # Jika file model belum ada di server, unduh dari Google Drive
    if not os.path.exists(output_model_path):
        with st.spinner('Sedang mengunduh model dari Google Drive, mohon tunggu...'):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, output_model_path, quiet=False)
            
    return tf.keras.models.load_model(output_model_path)

# ====================================================================
# PENTING: Ganti teks di bawah ini dengan ID File Google Drive Anda!
# ====================================================================
GOOGLE_DRIVE_FILE_ID = 'https://drive.google.com/file/d/1KEFv42CHGuDSwHuc9nbzsXexHZVTWxJY/view?usp=sharing'

# Memuat model menggunakan fungsi cache
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
