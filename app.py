import streamlit as st
import cv2
import numpy as np
import pickle
from skimage.feature import local_binary_pattern
from PIL import Image

# Fungsi untuk mengekstraksi fitur LBP
def extract_lbp_features(image, radius=1, n_points=8):
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist

# Fungsi untuk memuat model dan label encoder dari file pickle
def load_model():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    
    with open('label_encoder.pkl', 'rb') as le_file:
        label_encoder = pickle.load(le_file)
    
    return model, label_encoder

# Fungsi untuk mendeteksi dan mengenali wajah dalam gambar
def detect_and_recognize_face(image, face_cascade, model, label_encoder):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        features = extract_lbp_features(face_roi).reshape(1, -1)
        predicted_label = model.predict(features)
        predicted_person = label_encoder.inverse_transform(predicted_label)
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, predicted_person[0], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    return image

# Inisialisasi Streamlit
st.title("Pengenalan Wajah dengan LBP dan SVM")

# Upload gambar
uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert file image menjadi format yang bisa dibaca OpenCV
    img = Image.open(uploaded_file)
    img = np.array(img)

    # Muat model dan label encoder
    model, label_encoder = load_model()

    # Muat detektor wajah
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Lakukan prediksi pada gambar yang diupload
    result_image = detect_and_recognize_face(img, face_cascade, model, label_encoder)

    # Menampilkan gambar hasil deteksi dan prediksi
    st.image(result_image, caption="Hasil Prediksi", use_container_width=True)
