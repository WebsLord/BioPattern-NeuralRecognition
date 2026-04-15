import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split

IMG_SIZE = 128
DATASET_DIR = "DATASET"
CLASSES = ["cats", "dogs", "snakes"]
MAP = {0: "Cat", 1: "Dog", 2: "Snake"}
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data():
    X, y = [], []
    for index, category in enumerate(CLASSES):
        path = os.path.join(DATASET_DIR, category)
        if not os.path.exists(path): continue
        for img_name in tqdm(os.listdir(path), desc=f"Loading {category}"):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path)
                if img is None: continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                X.append(img)
                y.append(index)
            except: continue
    return np.array(X), np.array(y)

print("🚀 Loading Data...")
X, y = load_data()

# Verification Plot
plt.figure(figsize=(15, 6))
for i in range(10):
    idx = random.randint(0, len(X)-1)
    plt.subplot(2, 5, i+1)
    plt.imshow(X[idx])
    plt.title(f"Label: {MAP[y[idx]]}")
    plt.axis("off")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/data_prep_verification.png")
print(f"✅ Verification plot saved to {RESULTS_DIR}/data_prep_verification.png")

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

X_train, X_val, X_test = X_train/255.0, X_val/255.0, X_test/255.0

np.savez("preprocessed_data.npz", X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
print("✅ Data Preprocessed and Saved to preprocessed_data.npz")
