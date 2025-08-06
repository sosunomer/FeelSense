import cv2
import numpy as np
import os
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Sabit değişkenler
IMG_HEIGHT, IMG_WIDTH = 48, 48
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
FOLDER_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def load_data():
    """Eğitim ve test verilerini yükler"""
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    
    # Eğitim verilerini yükle
    print("Eğitim verilerini yükleme...")
    for i, folder in enumerate(FOLDER_LABELS):
        train_path = os.path.join('train', folder)
        for img_file in os.listdir(train_path)[:500]:  # Her sınıftan sadece 500 görüntü al
            img_path = os.path.join(train_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
                X_train.append(img.flatten())
                y_train.append(i)
    
    # Test verilerini yükle
    print("Test verilerini yükleme...")
    for i, folder in enumerate(FOLDER_LABELS):
        test_path = os.path.join('test', folder)
        for img_file in os.listdir(test_path)[:100]:  # Her sınıftan sadece 100 görüntü al
            img_path = os.path.join(test_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
                X_test.append(img.flatten())
                y_test.append(i)
    
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

def train_model():
    """SVM modelini eğitir ve kaydeder"""
    # Veriyi yükle
    X_train, y_train, X_test, y_test = load_data()
    
    # Veriyi normalize et
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    print(f"Eğitim veri boyutu: {X_train.shape}")
    print(f"Test veri boyutu: {X_test.shape}")
    
    # SVM modelini oluştur ve eğit
    print("Model eğitiliyor...")
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    
    # Test et
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test doğruluğu: {accuracy:.4f}")
    
    # Modeli kaydet
    print("Model kaydediliyor...")
    with open('emotion_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model başarıyla kaydedildi: emotion_model.pkl")
    
    return model, X_test, y_test

def plot_sample_predictions(model, X_test, y_test):
    """Örnek tahminleri görselleştirir"""
    plt.figure(figsize=(12, 8))
    
    # Rastgele 5 örnek seç
    indices = np.random.choice(len(X_test), 5, replace=False)
    
    for i, idx in enumerate(indices):
        # Görüntüyü al ve yeniden şekillendir
        img = X_test[idx].reshape(IMG_HEIGHT, IMG_WIDTH)
        
        # Tahmin yap
        pred = model.predict([X_test[idx]])[0]
        true_label = EMOTION_LABELS[y_test[idx]]
        pred_label = EMOTION_LABELS[pred]
        
        # Görüntüyü göster
        plt.subplot(1, 5, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Gerçek: {true_label}\nTahmin: {pred_label}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    plt.close()

if __name__ == "__main__":
    model, X_test, y_test = train_model()
    plot_sample_predictions(model, X_test, y_test)
    print("İşlem tamamlandı!") 