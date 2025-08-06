# FeelSense - Gerçek Zamanlı Duygu Tanıma Sistemi

Bu proje, kamera görüntülerinden yüz ifadelerini analiz ederek bireylerin duygularını (mutlu, üzgün, kızgın, şaşkın vb.) gerçek zamanlı olarak tahmin eden bir yapay zeka sistemidir. Makine öğrenmesi temelli SVM (Support Vector Machine) modeli kullanılarak geliştirilmiştir.

## Özellikler

- Kamera görüntüsünden yüz tespiti
- 7 farklı duygu durumu tanıma (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- Gerçek zamanlı duygu analizi
- Streamlit ile kullanıcı dostu arayüz
- Fotoğraf yükleme ve analiz etme seçeneği
- Sesli duygu bildirimi (opsiyonel)

## Kurulum

1. Gerekli paketleri yükleyin:

```bash
pip install -r requirements.txt
```

2. Modeli eğitin (opsiyonel, önceden eğitilmiş model mevcuttur):

```bash
python model_lite.py
```

## Kullanım

### Streamlit Arayüzü ile Kullanım

```bash
streamlit run app_lite.py
```

### Kamera ile Gerçek Zamanlı Duygu Tanıma

```bash
python camera_emotion_detection_lite.py
```

## Proje Yapısı

- `app_lite.py`: Streamlit web arayüzü
- `model_lite.py`: SVM modelini oluşturma ve eğitme
- `camera_emotion_detection_lite.py`: OpenCV ile gerçek zamanlı duygu tanıma
- `requirements.txt`: Gerekli kütüphaneler
- `emotion_model.pkl`: Eğitilmiş SVM modeli
- `sample_predictions.png`: Örnek tahmin sonuçları

## Veri Seti

Bu projede FER2013 veri seti kullanılmıştır. Veri seti, 48x48 piksel boyutunda gri tonlamalı yüz görüntülerinden oluşur ve 7 duygu sınıfı içerir:
- Angry (Kızgın)
- Disgust (İğrenme)
- Fear (Korku)
- Happy (Mutlu)
- Sad (Üzgün)
- Surprise (Şaşkın)
- Neutral (Nötr)

## Teknik Detaylar

- **Yüz Tespiti**: OpenCV'nin Haar Cascade sınıflandırıcısı
- **Duygu Tanıma Modeli**: Support Vector Machine (SVM)
- **Model Kernel**: Linear
- **Görüntü Boyutu**: 48x48 piksel
- **Sesli Bildirim**: pyttsx3 kütüphanesi

## Geliştirme Önerileri

- Daha karmaşık modeller denenebilir (Random Forest, Neural Networks)
- Daha fazla veri artırma tekniği uygulanabilir
- Farklı yüz tespit algoritmaları denenebilir
- Türkçe arayüz geliştirilebilir
- Model performansı iyileştirilebilir

## Lisans

MIT 