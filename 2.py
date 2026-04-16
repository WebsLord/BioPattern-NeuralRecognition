import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# --- STEP 1: Configurations ---
RESULTS_DIR = "results/mlp"
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

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

classes = ["Cats", "Dogs", "Snakes"]
input_dim = X_train_flat.shape[1]

# --- STEP 2: Model Builder ---
def build_mlp(config):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for units in config['hidden_layers']:
        model.add(layers.Dense(units, activation=config['activation']))
        if config.get('dropout'):
            model.add(layers.Dropout(config['dropout']))
    model.add(layers.Dense(3, activation='softmax'))
    opt = getattr(optimizers, config['optimizer'])(learning_rate=0.001)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

experiments = [
    {"name": "MLP_V1_Shallow", "hidden_layers": [128], "activation": "relu", "optimizer": "Adam"},
    {"name": "MLP_V2_Deep", "hidden_layers": [256, 128, 64], "activation": "relu", "optimizer": "Adam"},
    {"name": "MLP_V3_Tanh_RMS", "hidden_layers": [128, 128], "activation": "tanh", "optimizer": "RMSprop"},
    {"name": "MLP_V4_Wide_SGD", "hidden_layers": [512], "activation": "sigmoid", "optimizer": "SGD"},
    {"name": "MLP_V5_Optimized", "hidden_layers": [256, 128], "activation": "relu", "optimizer": "Adam", "dropout": 0.2}
]

results = []

# --- STEP 3: Training Loop ---
for config in experiments:
    print(f"\n🚀 Running: {config['name']}")
    model = build_mlp(config)
    history = model.fit(X_train_flat, y_train, validation_data=(X_val_flat, y_val), 
                        epochs=15, batch_size=32, verbose=1)
    
    # Visualizing History
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1); plt.plot(history.history['accuracy'], label='Train'); plt.plot(history.history['val_accuracy'], label='Val'); plt.legend(); plt.title(f"{config['name']} Acc")
    plt.subplot(1, 2, 2); plt.plot(history.history['loss'], label='Train'); plt.plot(history.history['val_loss'], label='Val'); plt.legend(); plt.title(f"{config['name']} Loss")
    plt.savefig(f"{RESULTS_DIR}/{config['name']}_history.png")
    plt.close()

    # Evaluation
    y_pred_probs = model.predict(X_test_flat)
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
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.title(f"{config['name']} Confusion Matrix")
    plt.savefig(f"{RESULTS_DIR}/{config['name']}_cm.png")
    plt.close()
    
    model.save(f"models/{config['name']}.h5")

# --- STEP 4: Save Summary ---
df = pd.DataFrame(results)
df.to_csv(f"{RESULTS_DIR}/mlp_results.csv", index=False)
print(f"\n📊 Results saved to {RESULTS_DIR}/mlp_results.csv")
