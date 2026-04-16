import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# --- SETTINGS / AYARLAR ---
# Ghost style: Managing our directories with precision.
# Ghost stili: Dizinlerimizi hassasiyetle yönetiyoruz.
RESULTS_DIR = "results/mlp"
MODEL_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Loading the preprocessed data we saved in Step 1.
# Adım 1'de kaydettiğimiz ön işlenmiş verileri yüklüyoruz.
print("📊 Loading preprocessed data... / Ön işlenmiş veriler yükleniyor...")
data = np.load("preprocessed_data.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]

# MLP requires flattened input, so we flatten our images.
# MLP düzleştirilmiş giriş gerektirir, bu yüzden görsellerimizi düzleştiriyoruz.
X_train_flat = X_train.reshape(X_train.shape[0], -1) / 255.0
X_val_flat = X_val.reshape(X_val.shape[0], -1) / 255.0
X_test_flat = X_test.reshape(X_test.shape[0], -1) / 255.0

# Define labels for our summary table / Özet tablomuz için etiketleri tanımlayalım
input_shape = X_train_flat.shape[1]
num_classes = 3
classes = ["Cats", "Dogs", "Snakes"]

def build_mlp(config):
    """
    Factory function to create different MLP architectures.
    Farklı MLP mimarileri oluşturmak için fabrika fonksiyonu.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    
    for units in config['hidden_units']:
        model.add(layers.Dense(units, activation=config['activation']))
        if config.get('dropout'):
            model.add(layers.Dropout(config['dropout']))
            
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    opt = getattr(optimizers, config['optimizer'])(learning_rate=0.001)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- EXPERIMENT CONFIGURATIONS / DENEY YAPILANDIRMALARI ---
experiments = [
    {"name": "MLP_V1_Shallow", "hidden_units": [128], "activation": "relu", "optimizer": "Adam"},
    {"name": "MLP_V2_Deep", "hidden_units": [256, 128, 64], "activation": "relu", "optimizer": "Adam"},
    {"name": "MLP_V3_Tanh_RMS", "hidden_units": [128, 128], "activation": "tanh", "optimizer": "RMSprop"},
    {"name": "MLP_V4_Wide_SGD", "hidden_units": [512], "activation": "sigmoid", "optimizer": "SGD"},
    {"name": "MLP_V5_Dropout", "hidden_units": [128, 64], "activation": "relu", "optimizer": "Adam", "dropout": 0.2}
]

summary_results = []

print(f"\n🧠 Starting MLP Experiments / MLP Deneyleri Başlıyor...")

for exp in experiments:
    print(f"\n--- 🚀 Running: {exp['name']} ---")
    model = build_mlp(exp)
    
    # Training (Ghost style tip: Keep epochs moderate for a midterm report comparison)
    # Eğitim (Ghost stil ipucu: Vize raporu karşılaştırması için epoch sayısını makul tutuyoruz)
    history = model.fit(
        X_train_flat, y_train,
        validation_data=(X_val_flat, y_val),
        epochs=15,
        batch_size=32,
        verbose=1
    )
    
    # Evaluation / Değerlendirme
    test_loss, test_acc = model.evaluate(X_test_flat, y_test, verbose=0)
    print(f"✅ {exp['name']} Test Accuracy: {test_acc:.4f}")
    
    summary_results.append({
        "Model": exp['name'],
        "Test Accuracy": test_acc,
        "Test Loss": test_loss,
        "Activation": exp['activation'],
        "Optimizer": exp['optimizer']
    })
    
    # Plotting training curves / Eğitim eğrilerini çizdirme
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
    y_pred = np.argmax(model.predict(X_test_flat), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f"{exp['name']} Confusion Matrix")
    plt.savefig(os.path.join(RESULTS_DIR, f"{exp['name']}_cm.png"))
    plt.close()
    
    # Save model / Modeli kaydet (Optional but good practice)
    model.save(os.path.join(MODEL_DIR, f"Efe_Yasar_{exp['name']}.keras"))

# Final summary table / Final özet tablosu
df_summary = pd.DataFrame(summary_results)
df_summary.to_csv(os.path.join(RESULTS_DIR, "mlp_comparison_summary.csv"), index=False)
print("\n📝 Comparison summary saved to mlp_comparison_summary.csv")

# Print nice table to console / Konsola güzel bir tablo basalım
print("\n" + "="*50)
print(df_summary.to_string(index=False))
print("="*50)
print("\n✨ Phase 3: MLP experiments completed successfully!")
