import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# --- STEP 1: Configurations ---
RESULTS_DIR = "results/cnn"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load data
if not os.path.exists("preprocessed_data.npz"):
    print("❌ Error: preprocessed_data.npz not found.")
    exit()

data = np.load("preprocessed_data.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]

classes = ["Cats", "Dogs", "Snakes"]

# --- STEP 2: CNN Builder ---
def build_cnn(config):
    model = models.Sequential()
    model.add(layers.Input(shape=(128, 128, 3)))
    for filters in config['layers']:
        model.add(layers.Conv2D(filters, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(config['dense_size'], activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

experiments = [
    {"name": "CNN_V1_Simple", "layers": [32, 64], "dense_size": 64},
    {"name": "CNN_V2_Deep", "layers": [32, 64, 128], "dense_size": 128},
    {"name": "CNN_V3_Wide", "layers": [64, 128], "dense_size": 256},
    {"name": "CNN_V4_Minimal", "layers": [16, 32], "dense_size": 32},
    {"name": "CNN_V5_Optimized", "layers": [32, 64, 128, 256], "dense_size": 256}
]

results = []

# --- STEP 3: Training Loop ---
for config in experiments:
    print(f"\n🚀 Running: {config['name']}")
    model = build_cnn(config)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                        epochs=15, batch_size=32, verbose=1)
    
    # Visualizing History
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1); plt.plot(history.history['accuracy'], label='Train'); plt.plot(history.history['val_accuracy'], label='Val'); plt.legend(); plt.title(f"{config['name']} Acc")
    plt.subplot(1, 2, 2); plt.plot(history.history['loss'], label='Train'); plt.plot(history.history['val_loss'], label='Val'); plt.legend(); plt.title(f"{config['name']} Loss")
    plt.savefig(f"{RESULTS_DIR}/{config['name']}_history.png")
    plt.close()

    # Evaluation
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    acc = np.mean(y_pred == y_test)
    
    results.append({
        "Model": config['name'],
        "Test Accuracy": acc,
        "Precision (Macro)": precision,
        "Recall (Macro)": recall,
        "F1-Score (Macro)": f1
    })
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Greens')
    plt.title(f"{config['name']} Confusion Matrix")
    plt.savefig(f"{RESULTS_DIR}/{config['name']}_cm.png")
    plt.close()
    
    model.save(f"models/{config['name']}.h5")

# --- STEP 4: Save Summary ---
df = pd.DataFrame(results)
df.to_csv(f"{RESULTS_DIR}/cnn_results.csv", index=False)
print(f"\n📊 Results saved to {RESULTS_DIR}/cnn_results.csv")
