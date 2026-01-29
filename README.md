🚗 DeepPark: Farklı Hava Koşullarında Otopark Doluluk Tespiti
Bu proje, PKLot veri seti kullanılarak, otopark alanlarındaki doluluk durumunu (Dolu/Boş) tespit etmek amacıyla geliştirilmiştir. Proje kapsamında 8 farklı derin öğrenme mimarisi (VGG16, DenseNet121, ResNet, MobileNet vb.) zorlu hava koşullarında (yağmurlu, bulutlu, gölgeli) test edilmiş ve karşılaştırılmıştır.

📋 İçindekiler
Proje Hakkında

Ekip ve Görev Dağılımı

Veri Seti ve Ön İşleme

Modeller ve Performans

Kurulum ve Çalıştırma

Proje Yapısı

📖 Proje Hakkında
Geleneksel sensör tabanlı sistemlerin aksine, bu proje kamera görüntüleri üzerinden Bilgisayarlı Görü (Computer Vision) tekniklerini kullanarak maliyet etkin bir çözüm sunar.

Temel Özellikler:

Hibrit Doğrulama: Modeller hem standart Split (%80-%20) hem de 5-Fold Cross Validation ile test edilmiştir.

Veri Zenginleştirme: Farklı açılara ve ışık koşullarına dayanıklılık için modele özgü Augmentation (Rotasyon, ColorJitter) uygulanmıştır.

Nihai Model: DenseNet121, %99.34 doğruluk ve yüksek kararlılık ile projenin "En İyi Modeli" seçilmiştir.

👥 Ekip ve Görev Dağılımı
İsim Soyad	Rol	Sorumlu Olduğu Modeller
Hamza Hakverir	Model Geliştirme & Raporlama	DenseNet121 (Best Model), VGG16
Abdulkadir	Veri İşleme & Model Eğitimi	ResNet18, MobileNetV3
Yunus Emre	Model Optimizasyonu	ShuffleNet V2, YOLOv8-cls
M. Emin	Test ve Analiz	Inception V3, EfficientNet B0
💾 Veri Seti ve Ön İşleme
Projede PKLot (Parking Lot Dataset) kullanılmıştır.

Kaynak: UFPR & PUCPR Kampüsleri (Brezilya).

Filtreleme: 12 Eylül 2012 tarihli, yağmurlu ve gölgeli görüntüler seçilmiştir.

Toplam Görüntü: 32.327 Adet (%57 Dolu, %43 Boş).

Ön İşleme:

Resize: 224x224 piksel.

Normalizasyon: ImageNet standartları.

Augmentation: RandomHorizontalFlip, RandomRotation (10-15 derece).

🏆 Modeller ve Performans
Tüm modeller aynı veri seti üzerinde eğitilmiş ve test edilmiştir. Aşağıdaki tablo, modellerin test seti üzerindeki performanslarını özetlemektedir.

M: Genel Model Performans Karşılaştırması
Model Adı	Öğrenci	Accuracy	F1-Score	Hata Sayısı
ShuffleNet V2	Yunus Emre Edizer	%99.96	0.999	2
YOLOv8-cls	Yunus Emre Edizer	%99.94	0.999	~4
ResNet18	Abdulkadir Gedik	%99.92	0.9992	~5
InceptionV3	M. Emin Çapan	%99.86	0.9986	~9
MobileNetV3	Abdulkadir Gedik	%99.80	0.9980	~13
EfficientNet-B0	M. Emin Çapan	%99.64	0.9964	~23
DenseNet121	Hamza Hakverir	%99.34	0.9934	33
VGG16	Hamza Hakverir	%99.10	0.9910	~58


⚙️ Kurulum ve Çalıştırma
Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin:

1. Repoyu Klonlayın:

Bash
git clone https://github.com/KullaniciAdiniz/DeepPark.git
cd DeepPark
2. Gerekli Kütüphaneleri Yükleyin:

Bash
pip install -r requirements.txt
(Gereksinimler: torch, torchvision, pandas, numpy, matplotlib, seaborn, scikit-learn)

3. Veri Setini Hazırlayın: PKLot veri setini indirin ve data/ klasörüne çıkartın. Klasör yapısı şöyle olmalıdır:

data/
  ├── empty/
  └── occupied/
4. Eğitimi Başlatın (Örnek: VGG16):

Bash
python notebooks/VGG16_Egitim.ipynb
📂 Proje Yapısı
Bash
DeepPark/
├── data/                  # Veri seti (Git'e yüklenmez, yerelde tutulur)
├── docs/                  # Raporlar ve Sunum Dosyaları
│   ├── Final_Raporu.pdf
│   └── Sunum.pdf
├── models/                # Eğitilmiş model ağırlıkları (.pth dosyaları)
├── notebooks/             # Jupyter Notebook kodları
│   ├── DenseNet_Training.ipynb
│   ├── VGG16_Training.ipynb
│   └── ...
├── results/               # Confusion Matrix ve Grafik çıktıları
├── README.md              # Proje dokümantasyonu
└── requirements.txt       # Kütüphane listesi
🤝 Katkıda Bulunma (Git Kuralları)
Ana dal (main) üzerinde doğrudan değişiklik yapmayın.

Her yeni özellik veya düzeltme için yeni bir dal (branch) açın: git checkout -b feature/yeni-ozellik.

Commit mesajlarınızı açıklayıcı yazın: git commit -m "VGG16 eğitim grafikleri eklendi".

Değişiklikleri göndermeden önce mutlaka git pull yapın.

DeepPark Ekibi © 2026
