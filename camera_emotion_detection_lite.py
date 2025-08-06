import cv2
import numpy as np
import pickle
import time

# Sabit değişkenler
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
FOLDER_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
MODEL_PATH = 'emotion_model.pkl'
FACE_CLASSIFIER_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
IMG_HEIGHT, IMG_WIDTH = 48, 48

def load_emotion_model():
    """Eğitilmiş duygu tanıma modelini yükler"""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("Model başarıyla yüklendi!")
        return model
    except Exception as e:
        print(f"Model yüklenirken hata oluştu: {e}")
        return None

def detect_emotions():
    """Kamera görüntüsünden duygu tanıma yapar"""
    # Yüz sınıflandırıcısını yükle
    face_classifier = cv2.CascadeClassifier(FACE_CLASSIFIER_PATH)
    
    # Duygu tanıma modelini yükle
    emotion_model = load_emotion_model()
    if emotion_model is None:
        print("Model yüklenemedi. Program sonlandırılıyor...")
        return
    
    # Kamerayı başlat
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılamadı!")
        return
    
    print("Duygu tanıma başlatıldı. Çıkmak için 'q' tuşuna basın.")
    
    # FPS hesaplama için değişkenler
    prev_time = time.time()
    fps = 0
    fps_counter = 0
    fps_update_interval = 10  # Her 10 karede bir FPS güncellenir
    
    while True:
        # Kameradan kare oku
        ret, frame = cap.read()
        if not ret:
            print("Kamera görüntüsü alınamadı!")
            break
        
        # FPS hesapla
        fps_counter += 1
        if fps_counter >= fps_update_interval:
            current_time = time.time()
            fps = fps_update_interval / (current_time - prev_time)
            prev_time = current_time
            fps_counter = 0
        
        # Görüntüyü gri tonlamaya dönüştür
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Yüzleri tespit et
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
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
                confidence = prediction[emotion_idx] * 100
                
                # Sonuçları görüntüye ekle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Duygu etiketi ve güven değeri
                label_text = f"{emotion_label}: {confidence:.1f}%"
                cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            except Exception as e:
                print(f"Tahmin hatası: {e}")
        
        # FPS değerini ekrana yaz
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Sonuçları göster
        cv2.imshow('Duygu Tanıma', frame)
        
        # Çıkış için 'q' tuşunu kontrol et
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Kaynakları serbest bırak
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_emotions() 