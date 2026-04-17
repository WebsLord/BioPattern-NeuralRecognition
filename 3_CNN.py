import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- SETTINGS / AYARLAR ---
RESULTS_DIR = "results/cnn"
MODEL_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Loading data / Veriyi yüklüyoruz
print("📊 Loading preprocessed data for CNN... / CNN için ön işlenmiş veriler yükleniyor...")
data = np.load("preprocessed_data.npz")
X_train, y_train = data["X_train"] / 255.0, data["y_train"]
X_val, y_val = data["X_val"] / 255.0, data["y_val"]
X_test, y_test = data["X_test"] / 255.0, data["y_test"]

input_shape = (128, 128, 3)
num_classes = 3
classes = ["Cats", "Dogs", "Snakes"]

def build_cnn(config):
    """
    Constructs a CNN based on the configuration provided.
    Verilen yapılandırmaya göre bir CNN oluşturur.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    
    # Adding Convolutional Layers / Konvolüsyon Katmanlarını Ekliyoruz
    for filters in config['filters']:
        model.add(layers.Conv2D(filters, config['kernel_size'], padding='same', activation='relu'))
        if config.get('batch_norm'):
            model.add(layers.BatchNormalization())
        
        if config['pooling'] == 'max':
            model.add(layers.MaxPooling2D((2, 2)))
        elif config['pooling'] == 'avg':
            model.add(layers.AveragePooling2D((2, 2)))
            
    model.add(layers.Flatten())
    
    # Fully Connected Layers / Tam Bağlantılı Katmanlar
    for units in config['dense_units']:
        model.add(layers.Dense(units, activation='relu'))
        if config.get('dropout'):
            model.add(layers.Dropout(config['dropout']))
            
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- CNN EXPERIMENT CONFIGURATIONS / CNN DENEY YAPILANDIRMALARI ---
experiments = [
    {"name": "CNN_V1_Base", "filters": [32, 64], "kernel_size": (3,3), "pooling": "max", "dense_units": [128]},
    {"name": "CNN_V2_LargeKernel", "filters": [32, 64], "kernel_size": (5,5), "pooling": "max", "dense_units": [128]},
    {"name": "CNN_V3_Deep", "filters": [32, 64, 128, 256], "kernel_size": (3,3), "pooling": "max", "dense_units": [256]},
    {"name": "CNN_V4_AvgPool", "filters": [32, 64], "kernel_size": (3,3), "pooling": "avg", "dense_units": [128]},
    {"name": "CNN_V5_Optimized", "filters": [32, 64, 128], "kernel_size": (3,3), "pooling": "max", "dense_units": [128], "batch_norm": True, "dropout": 0.3}
]

cnn_summary = []

print(f"\n🖼️ Starting CNN Experiments / CNN Deneyleri Başlıyor...")

for exp in experiments:
    print(f"\n--- 🚀 Running: {exp['name']} ---")
    model = build_cnn(exp)
    
    # Ghost style: Training with validation to catch overfitting early.
    # Ghost stili: Aşırı öğrenmeyi erkenden yakalamak için doğrulama ile eğitim.
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10, # CNN is complex, 10 epochs is a good start for comparison.
        batch_size=32,
        verbose=1
    )
    
    # Evaluation / Değerlendirme
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"✅ {exp['name']} Test Accuracy: {test_acc:.4f}")
    
    cnn_summary.append({
        "Model": exp['name'],
        "Test Accuracy": test_acc,
        "Test Loss": test_loss,
        "Kernel Size": str(exp['kernel_size']),
        "Pooling": exp['pooling']
    })
    
    # Plotting / Grafik Çizdirme
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f"{exp['name']} Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f"{exp['name']} Loss")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f"{exp['name']}_history.png"))
    plt.close()
    
    # Confusion Matrix / Karmaşıklık Matrisi
    y_pred = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Greens')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f"{exp['name']} Confusion Matrix")
    plt.savefig(os.path.join(RESULTS_DIR, f"{exp['name']}_cm.png"))
    plt.close()
    
    model.save(os.path.join(MODEL_DIR, f"Efe_Yasar_{exp['name']}.keras"))

# Summary table / Özet tablo
df_cnn = pd.DataFrame(cnn_summary)
df_cnn.to_csv(os.path.join(RESULTS_DIR, "cnn_comparison_summary.csv"), index=False)
print("\n📝 Comparison summary saved to cnn_comparison_summary.csv")

print("\n" + "="*50)
print(df_cnn.to_string(index=False))
print("="*50)
print("\n✨ Phase 4: CNN experiments completed successfully!")
