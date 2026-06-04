import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

model = tf.keras.models.load_model('model_hewan.keras')
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
