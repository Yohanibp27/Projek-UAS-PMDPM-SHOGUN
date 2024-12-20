# dikerjakan oleh: Yohani Seprini (210711478)

import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model(r"BestModel_VGG16_Shogun.h5")
class_names = ['Onion', 'Red_Onion', 'Garlic']

st.markdown("""
    <style>
        body {
            background-color: #F5F5F5; /* Light grey for modern theme */
            color: #333333; /* Dark grey text */
            font-family: "Arial", sans-serif;
        }
        .title {
            color: #0C0C0C;
            text-align: center;
            margin-bottom: 30px;
        }
        .prediction-box {
            border: 2px solid #9E9E9E;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            background-color: #FFFFFF;
        }
        .upload-container {
            border: 1px dashed #0C0C0C;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        .file-name {
            font-weight: bold;
            color: #333333;
        }
    </style>
""", unsafe_allow_html=True)

def classify_image(image):
    try:
        input_image = image.resize((224, 224)) 
        input_image_array = np.array(input_image) / 255.0 
        input_image_array_exp_dim = np.expand_dims(input_image_array, axis=0)

        predictions = model.predict(input_image_array_exp_dim)
        result = tf.nn.softmax(predictions[0])

        class_idx = np.argmax(result)
        confidence_scores = result.numpy()
        return class_names[class_idx], confidence_scores
    except Exception as e:
        return "Error", str(e)

def custom_progress_bar(confidence, colors):
    progress_html = f"""
    <div style="border: 1px solid #ddd; border-radius: 5px; overflow: hidden; width: 100%; font-size: 14px; display: flex;">
        <div style="width: {confidence[0] * 100:.2f}%; background: {colors[0]}; color: white; text-align: center; height: 24px;">
            {confidence[0] * 100:.2f}%
        </div>
        <div style="width: {confidence[1] * 100:.2f}%; background: {colors[1]}; color: white; text-align: center; height: 24px;">
            {confidence[1] * 100:.2f}%
        </div>
        <div style="width: {confidence[2] * 100:.2f}%; background: {colors[2]}; color: white; text-align: center; height: 24px;">
            {confidence[2] * 100:.2f}%
        </div>
    </div>
    """
    st.markdown(progress_html, unsafe_allow_html=True)

st.markdown("<h1 class='title'>🧅 Prediksi Jenis Bawang - Kelompok Shogun</h1>", unsafe_allow_html=True)
st.markdown("""
<div class="upload-container">
    <p>📤 <strong>Unggah Gambar</strong> (format: jpg, jpeg, png)</p>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader("", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if st.button("🔍 **Prediksi**"):
    if uploaded_files:
        st.write("### 📝 Hasil Prediksi:")
        for uploaded_file in uploaded_files:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption=uploaded_file.name, use_column_width=True)
                label, confidence = classify_image(image)
                if label != "Error":
                    colors = ["#FFCA28", "#E53935", "#007BFF"]
                    st.markdown(f"""
                    <div class='prediction-box'>
                        <p class='file-name'>🖼️ <strong>File Name:</strong> {uploaded_file.name}</p>
                        <h4 style='color: {colors[class_names.index(label)]};'>🔮 <strong>Prediction:</strong> {label}</h4>
                        <p><strong>Confidence Scores:</strong></p>
                        <ul>
                            <li>Onion: {confidence[0] * 100:.2f}%</li>
                            <li>Red Onion: {confidence[1] * 100:.2f}%</li>
                            <li>Garlic: {confidence[2] * 100:.2f}%</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    custom_progress_bar(confidence, colors)
                else:
                    st.write(f"❌ Kesalahan saat memproses gambar: {uploaded_file.name}: {confidence}")
            except Exception as e:
                st.write(f"❌ Error: Tidak dapat memproses file {uploaded_file.name} ({str(e)})")
    else:
        st.write("⚠️ Silakan unggah setidaknya satu gambar untuk diprediksi.")
