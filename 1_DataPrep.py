import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tqdm import tqdm

# --- CONFIGURATION / YAPILANDIRMA ---
# We are setting a standard size for our animal images. 128x128 is a sweet spot for quality vs performance.
# Hayvan görsellerimiz için standart bir boyut belirliyoruz. 128x128 kalite ve performans için ideal bir denge noktasıdır.
IMG_SIZE = 128
DATASET_DIR = "DATASET"
CLASSES = ["cats", "dogs", "snakes"]
OUTPUT_DIR = "results/preprocessing"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOADING THE DATASET / VERİ SETİNİN YÜKLENMESİ ---
# We iterate through the folders and load images, ensuring we handle any corrupted files gracefully.
# Klasörler arasında gezinerek görselleri yüklüyoruz ve bozuk dosyaları güvenli bir şekilde atlıyoruz.

def load_data():
    X = []
    y = []
    
    print("🚀 Starting dataset loading... / Veri seti yükleme başlıyor...")
    for index, category in enumerate(CLASSES):
        path = os.path.join(DATASET_DIR, category)
        print(f"📂 Processing category: {category} / {category} sınıfı işleniyor...")
        
        for img_name in tqdm(os.listdir(path)):
            try:
                img_path = os.path.join(path, img_name)
                # Load image and resize / Görseli yükle ve boyutlandır
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                
                X.append(img)
                y.append(index)
            except Exception as e:
                # Engineering note: Keeping it quiet but logging for stability.
                # Mühendislik notu: İstikrar için hataları sessizce topluyoruz.
                pass
                
    return np.array(X), np.array(y)

X, y = load_data()
print(f"✅ Loaded {len(X)} images. / {len(X)} görsel yüklendi.")

# --- DATA SPLITTING / VERİYİ BÖLMELERE AYIRMAK ---
# We split into 70% Train, 15% Validation, and 15% Test.
# Veriyi %70 Eğitim, %15 Doğrulama ve %15 Test olarak ayırıyoruz.

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"📊 Training: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")

# --- DATA AUGMENTATION / VERİ ARTTIRMA ---
# To prevent overfitting, we introduce variety into our training data through random transformations.
# Aşırı öğrenmeyi önlemek için eğitim verilerimize rastgele dönüşümler ekliyoruz.

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Visualize some augmented images / Bazı arttırılmış görselleri görselleştirelim
plt.figure(figsize=(12, 8))
for i in range(6):
    plt.subplot(2, 3, i+1)
    img_array = X_train[i].reshape((1, IMG_SIZE, IMG_SIZE, 3))
    # Generate batch of 1 augmented image / 1 adet arttırılmış görsel oluştur
    augmented_iter = datagen.flow(img_array, batch_size=1)
    aug_img = next(augmented_iter)[0].astype('uint8')
    plt.imshow(aug_img)
    plt.title(f"Augmented: {CLASSES[y_train[i]]}")
    plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "augmented_samples.png"))
print(f"🖼️ Sample augmented images saved to {OUTPUT_DIR}/augmented_samples.png")

# --- SAVING PREPROCESSED DATA (OPTIONAL) / ÖN İŞLENMİŞ VERİYİ KAYDETME ---
# Engineering decision: For faster iterations, we can save the arrays, though for massive datasets it's better to use generators.
# Mühendislik kararı: Daha hızlı iterasyonlar için dizileri kaydedebiliriz, ancak devasa veri setlerinde jeneratör kullanmak daha iyidir.

# np.savez_compressed(os.path.join(DATASET_DIR, "preprocessed_data.npz"), 
#                     X_train=X_train, y_train=y_train, 
#                     X_val=X_val, y_val=y_val, 
#                     X_test=X_test, y_test=y_test)

print("\n✨ Phase 1 Complete. Dataset is ready for Neural Network architectures!")
print("✨ Faz 1 Tamamlandı. Veri seti Sinir Ağı mimarileri için hazır!")
