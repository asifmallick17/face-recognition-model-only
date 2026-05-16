import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# Load data
# -------------------------------
embeddings = np.load("embeddings.npy")
labels = np.load("labels.npy")

print("Embeddings shape:", embeddings.shape)
print("Unique persons:", np.unique(labels))

# -------------------------------
# Encode labels
# -------------------------------
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)

# -------------------------------
# Normalize embeddings
# -------------------------------
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

# -------------------------------
# Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    embeddings_scaled,
    labels_encoded,
    test_size=0.2,
    stratify=labels_encoded,
    random_state=42
)

# -------------------------------
# Final SVM Model (BEST PARAMS)
# -------------------------------
model = SVC(
    kernel='rbf',
    C=1,
    gamma='scale',
    probability=True,
    class_weight='balanced'
)

model.fit(X_train, y_train)

# -------------------------------
# Evaluate
# -------------------------------
y_pred = model.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# -------------------------------
# Save model
# -------------------------------
with open("face_recognition_model.pkl", "wb") as f:
    pickle.dump({
        "scaler": scaler,
        "svm": model,
        "encoder": encoder
    }, f)

print("\n✅ Model saved successfully!")