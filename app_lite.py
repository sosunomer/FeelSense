import streamlit as st
import cv2
import numpy as np
import pickle
import time
from PIL import Image
import io
try:
    import pyttsx3
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
import threading

# Sabit değişkenler
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
FOLDER_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
MODEL_PATH = 'emotion_model.pkl'
FACE_CLASSIFIER_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
IMG_HEIGHT, IMG_WIDTH = 48, 48

# Duygu etiketlerinin Türkçe karşılıkları
EMOTION_LABELS_TR = {
    'Angry': 'Kızgın',
    'Disgust': 'İğrenme',
    'Fear': 'Korku',
    'Happy': 'Mutlu',
    'Sad': 'Üzgün',
    'Surprise': 'Şaşkın',
    'Neutral': 'Nötr'
}

# Ses motoru başlatma
def initialize_speech_engine():
    if not SPEECH_AVAILABLE:
        return None
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Konuşma hızı
        engine.setProperty('volume', 1.0)  # Ses seviyesi
        return engine
    except Exception as e:
        print(f"Ses motoru başlatılamadı: {e}")
        return None

# Sesli duygu bildirimi
def speak_emotion(emotion, engine):
    if engine is None:
        return
    
    def speak_thread():
        try:
            # Türkçe karşılığını bul
            emotion_tr = EMOTION_LABELS_TR.get(emotion, emotion)
            engine.say(f"Tespit edilen duygu: {emotion_tr}")
            engine.runAndWait()
        except Exception as e:
            print(f"Sesli bildirim hatası: {e}")
    
    # Konuşmayı ayrı bir thread'de çalıştır
    threading.Thread(target=speak_thread).start()

# Streamlit sayfa yapılandırması
st.set_page_config(
    page_title="FeelSense - Emotion Recognition",
    page_icon="😊",
    layout="wide"
)

# Başlık ve açıklama
st.title("FeelSense - Real-Time Emotion Recognition")
st.markdown("""
This application analyzes facial expressions from camera images to predict emotional states.
""")

# Yan panel ayarları
with st.sidebar:
    st.header("Settings")
    detection_confidence = st.slider("Detection Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    st.divider()
    st.subheader("About")
    st.info("This application is developed using Scikit-learn and OpenCV.")
    st.markdown("Supported emotions:")
    for emotion in EMOTION_LABELS:
        st.markdown(f"- {emotion}")

@st.cache_resource
def load_emotion_model():
    """Eğitilmiş duygu tanıma modelini yükler"""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {e}")
        return None

def process_image(image, face_classifier, emotion_model, speech_engine=None, speak=False):
    """Görüntüyü işleyip duygu tanıma yapar"""
    # Görüntüyü OpenCV formatına dönüştür
    img_array = np.array(image)
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Görüntüyü gri tonlamaya dönüştür
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Yüzleri tespit et
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    results = []
    
    # Tespit edilen her yüz için duygu tanıma yap
    for (x, y, w, h) in faces:
        # Yüz bölgesini kırp
        roi_gray = gray[y:y+h, x:x+w]
        
        # Modele uygun boyuta getir ve normalize et
        roi = cv2.resize(roi_gray, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
        roi = roi.flatten().astype('float') / 255.0
        
        # Duygu tahmini yap
        try:
            prediction = emotion_model.predict_proba([roi])[0]
            emotion_idx = np.argmax(prediction)
            emotion_label = EMOTION_LABELS[emotion_idx]
            confidence = prediction[emotion_idx]
            
            # Sonuçları görüntüye ekle
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Duygu etiketi ve güven değeri
            label_text = f"{emotion_label}: {confidence:.1f}%"
            cv2.putText(img, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Sonuçları listeye ekle
            results.append({
                "emotion": emotion_label,
                "confidence": confidence,
                "position": (x, y, w, h)
            })
            
            # Sesli bildirim
            if speak and speech_engine and confidence > 0.5:  # Sadece güven değeri yüksek olanları seslendir
                speak_emotion(emotion_label, speech_engine)
                
        except Exception as e:
            st.error(f"Tahmin hatası: {e}")
    
    # Görüntüyü RGB formatına dönüştür (Streamlit için)
    result_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return result_img, results

def main():
    # Yüz sınıflandırıcısını yükle
    face_classifier = cv2.CascadeClassifier(FACE_CLASSIFIER_PATH)
    
    # Duygu tanıma modelini yükle
    emotion_model = load_emotion_model()
    if emotion_model is None:
        st.error("Model could not be loaded. Please check if the model file is in the correct location.")
        return
    
    # Ses motorunu başlat
    speech_engine = initialize_speech_engine()
    if speech_engine is None and SPEECH_AVAILABLE:
        st.info("Sesli bildirim özelliği bu platformda kullanılamıyor.")
    elif not SPEECH_AVAILABLE:
        st.info("Ses kütüphanesi yüklü değil. Uygulama sessiz modda çalışacak.")
    
    # Sekme seçenekleri
    tab1, tab2 = st.tabs(["📷 Camera", "🖼️ Upload Photo"])
    
    with tab1:
        st.header("Emotion Recognition with Camera")
        
        # Kamera kontrolü için session state kullan
        if 'camera_on' not in st.session_state:
            st.session_state.camera_on = False
        if 'frame_count' not in st.session_state:
            st.session_state.frame_count = 0
        
        # Sesli bildirim seçeneği
        enable_voice = st.checkbox("Enable voice feedback", value=True)
        
        # Kamera başlatma/durdurma butonları
        col1, col2 = st.columns(2)
        
        with col1:
            if not st.session_state.camera_on:
                start_button = st.button("Start Camera", type="primary", use_container_width=True, key="start_btn")
                if start_button:
                    st.session_state.camera_on = True
                    st.session_state.frame_count = 0
                    st.rerun()
        
        with col2:
            if st.session_state.camera_on:
                stop_button = st.button("Stop Camera", type="secondary", use_container_width=True, key="stop_btn")
                if stop_button:
                    st.session_state.camera_on = False
                    st.rerun()
        
        # Kamera görüntüsü için yer tutucu
        camera_container = st.container()
        result_container = st.container()
        
        # Kamera açıksa görüntü akışını başlat
        if st.session_state.camera_on:
            with camera_container:
                st.info("📹 Camera is active. Click 'Stop Camera' to exit.")
                
                # Placeholder'ları container içinde oluştur
                camera_placeholder = st.empty()
                
            with result_container:
                result_placeholder = st.empty()
            
            # Webcam widget kullanımı (daha stabil)
            try:
                # Basit kamera görüntüsü alma
                img_file_buffer = st.camera_input("Take a picture for emotion analysis", key="camera_input")
                
                if img_file_buffer is not None:
                    # Görüntüyü işle
                    image = Image.open(img_file_buffer)
                    result_img, results = process_image(
                        image, 
                        face_classifier, 
                        emotion_model,
                        speech_engine if enable_voice else None,
                        enable_voice
                    )
                    
                    # Sonuçları göster
                    with camera_container:
                        st.image(result_img, caption="Analysis Result", width=640)
                    
                    with result_container:
                        if results:
                            st.success("✅ Faces detected!")
                            for i, result in enumerate(results):
                                st.write(f"Face {i+1}: **{result['emotion']}** ({result['confidence']:.1%} confidence)")
                        else:
                            st.warning("No faces detected in the image.")
                            
            except Exception as e:
                st.error(f"Camera error: {str(e)}")
                st.session_state.camera_on = False
        else:
            with camera_container:
                st.info("📸 Click 'Start Camera' to activate emotion recognition.")
    
    with tab2:
        st.header("Emotion Recognition with Photo")
        
        # Sesli bildirim seçeneği
        enable_voice_photo = st.checkbox("Enable voice feedback for photo", value=True)
        
        # Dosya yükleme
        uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Yüklenen görüntüyü oku
            image = Image.open(uploaded_file)
            
            # Görüntüyü göster
            st.image(image, caption="Uploaded photo", width=640)
            
            # Görüntüyü işle
            if st.button("Analyze Emotions", type="primary"):
                with st.spinner("Analyzing..."):
                    result_img, results = process_image(
                        image, 
                        face_classifier, 
                        emotion_model,
                        speech_engine if enable_voice_photo else None,
                        enable_voice_photo
                    )
                    
                    # İşlenmiş görüntüyü göster
                    st.image(result_img, caption="Analysis result", width=640)
                    
                    # Duygu sonuçlarını göster
                    if results:
                        st.subheader("Detected emotions:")
                        for i, result in enumerate(results):
                            st.write(f"Face {i+1}: {result['emotion']} ({result['confidence']:.1%} confidence)")
                    else:
                        st.warning("No faces detected in the photo!")

if __name__ == "__main__":
    main() 