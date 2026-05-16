import pickle
import numpy as np
import cv2
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Load saved model
# -------------------------------
with open("face_recognition_model.pkl", "rb") as f:
    data = pickle.load(f)

scaler = data["scaler"]
model = data["svm"]
encoder = data["encoder"]

print("Model loaded!")

# -------------------------------
# Build centroid database (IMPORTANT)
# -------------------------------
embeddings = np.load("embeddings.npy")
labels = np.load("labels.npy")

database = {}
for name in np.unique(labels):
    database[name] = np.mean(embeddings[labels == name], axis=0)

print("Database ready!")

# -------------------------------
# Initialize FaceNet + MTCNN
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(image_size=160, margin=0, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# -------------------------------
# Recognition
# -------------------------------
def recognize_faces(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("Image not found!")
        return []

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    boxes, _ = mtcnn.detect(img_pil)

    if boxes is None:
        print("No faces detected")
        return []

    h, w, _ = img.shape
    present_people = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)

        # Safe bounding box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        face = img_rgb[y1:y2, x1:x2]

        if face.size == 0:
            continue

        face_pil = Image.fromarray(face)
        face_tensor = mtcnn(face_pil)

        if face_tensor is None:
            continue

        face_tensor = face_tensor.unsqueeze(0).to(device)

        # -------------------------------
        # Get embedding
        # -------------------------------
        with torch.no_grad():
            embedding = facenet(face_tensor).cpu().numpy()[0]

        # -------------------------------
        # SVM Prediction
        # -------------------------------
        embedding_scaled = scaler.transform(embedding.reshape(1, -1))

        proba = model.predict_proba(embedding_scaled)
        confidence = np.max(proba)
        pred = model.predict(embedding_scaled)

        svm_name = encoder.inverse_transform(pred)[0]

        # -------------------------------
        # Cosine Similarity (centroid-based)
        # -------------------------------
        best_name = "Unknown"
        best_score = -1

        for name, db_emb in database.items():
            score = cosine_similarity([embedding], [db_emb])[0][0]

            if score > best_score:
                best_score = score
                best_name = name

        # -------------------------------
        # Final Decision (combined)
        # -------------------------------
        if confidence > 0.7 and best_score > 0.6:
            name = svm_name
        else:
            name = "Unknown"

        present_people.append(name)

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img,
                    f"{name} ({confidence:.2f}/{best_score:.2f})",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2)

    present_people = list(set(present_people))

    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return present_people


# -------------------------------
# Test
# -------------------------------
result = recognize_faces("TestImage2.jpeg")
print("Present:", result)