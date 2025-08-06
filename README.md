# FeelSense - Gerçek Zamanlı Duygu Tanıma Sistemi 😊

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.22+-red.svg)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.7+-green.svg)](https://opencv.org)

Bu proje, kamera görüntülerinden yüz ifadelerini analiz ederek bireylerin duygularını (mutlu, üzgün, kızgın, şaşkın vb.) gerçek zamanlı olarak tahmin eden bir yapay zeka sistemidir. Makine öğrenmesi temelli SVM (Support Vector Machine) modeli kullanılarak geliştirilmiştir.

## 🎯 Demo

![FeelSense Demo](sample_predictions.png)

> **Canlı Demo**: Projeyi yerel ortamınızda çalıştırarak gerçek zamanlı duygu tanımayı deneyimleyebilirsiniz!

## ✨ Özellikler

- 📹 **Gerçek Zamanlı Analiz**: Kamera görüntüsünden anlık yüz tespiti ve duygu analizi
- 🎭 **7 Duygu Kategorisi**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- 🚀 **Hızlı ve Etkili**: SVM modeli ile optimize edilmiş performans
- 🌐 **Web Arayüzü**: Streamlit ile kullanıcı dostu, modern arayüz
- 📸 **Fotoğraf Analizi**: Yüklenen fotoğraflardan duygu tanıma
- 🔊 **Sesli Bildirim**: Opsiyonel ses ile duygu bildirimi
- ⚡ **Hafif Yapı**: Minimal kütüphane bağımlılığı ile hızlı kurulum

## 🚀 Hızlı Başlangıç

### 1. Repository'yi Klonlayın
```bash
git clone https://github.com/sosunomer/FeelSense.git
cd FeelSense
```

### 2. Gerekli Paketleri Yükleyin
```bash
pip install -r requirements.txt
```

### 3. Uygulamayı Çalıştırın
```bash
# Web arayüzü ile (Önerilen)
streamlit run app_lite.py

# Veya terminal tabanlı kamera kullanımı
python camera_emotion_detection_lite.py
```

### 4. Tarayıcıda Açın
http://localhost:8501 adresine gidin ve duygu tanımayı deneyimleyin!

## 📖 Kullanım Kılavuzu

### Web Arayüzü
1. **Kamera Sekmesi**: "Start Camera" ile gerçek zamanlı analiz başlatın
2. **Fotoğraf Sekmesi**: Bilgisayarınızdan fotoğraf yükleyip analiz edin
3. **Ayarlar**: Yan panelden tespit hassasiyetini ayarlayın

## 📁 Proje Yapısı

```
FeelSense/
├── 🌐 app_lite.py                    # Streamlit web arayüzü
├── 🤖 model_lite.py                  # SVM model eğitimi
├── 📹 camera_emotion_detection_lite.py # Terminal tabanlı kamera kullanımı
├── 📦 requirements.txt               # Gerekli kütüphaneler
├── 🧠 emotion_model.pkl             # Eğitilmiş SVM modeli
├── 📸 sample_predictions.png         # Örnek sonuçlar
└── 📋 README.md                     # Proje dokümantasyonu
```

## 📊 Duygu Kategorileri

| Duygu | İngilizce | Açıklama |
|-------|-----------|----------|
| 😠 | Angry | Kızgınlık, öfke durumu |
| 🤢 | Disgust | İğrenme, tiksinti |
| 😨 | Fear | Korku, endişe |
| 😊 | Happy | Mutluluk, sevinç |
| 😢 | Sad | Üzüntü, keder |
| 😲 | Surprise | Şaşkınlık, hayret |
| 😐 | Neutral | Nötr, doğal ifade |

## ⚙️ Teknik Detaylar

- **🔍 Yüz Tespiti**: OpenCV Haar Cascade Classifier
- **🧠 Model**: Support Vector Machine (SVM) - Linear Kernel
- **📏 Görüntü Boyutu**: 48x48 piksel (gri tonlama)
- **📁 Veri Seti**: FER2013 (Facial Expression Recognition)
- **🔊 Ses Motoru**: pyttsx3 (opsiyonel)
- **⚡ Performans**: Gerçek zamanlı işleme (~10 FPS)

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Branch'inizi push edin (`git push origin feature/AmazingFeature`)
5. Pull Request oluşturun

## 🚀 Gelecek Geliştirmeler

- [ ] Daha fazla duygu kategorisi
- [ ] Deep Learning modeli entegrasyonu
- [ ] Türkçe arayüz seçeneği
- [ ] Batch processing özelliği
- [ ] REST API desteği
- [ ] Docker containerization

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakınız.

## 🙏 Teşekkürler

- FER2013 veri seti sağlayıcıları
- OpenCV ve Streamlit toplulukları
- Scikit-learn geliştiricileri

---

⭐ **Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!** ⭐ 