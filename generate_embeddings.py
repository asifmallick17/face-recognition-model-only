import os
import numpy as np
import cv2
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# -------------------------------
# Initialize models
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(image_size=160, margin=0, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# -------------------------------
# Dataset path
# -------------------------------
dataset_path = "datasets"

all_embeddings = []
all_labels = []

# -------------------------------
# Loop through all folders (persons)
# -------------------------------
for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_path):
        continue

    print(f"\nProcessing person: {person_name}")

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Detect + align face
        face = mtcnn(img_pil)

        if face is None:
            print(f"❌ Face not detected: {img_name}")
            continue

        face = face.unsqueeze(0).to(device)

        # Get embedding
        with torch.no_grad():
            embedding = facenet(face).cpu().numpy()[0]

        all_embeddings.append(embedding)
        all_labels.append(person_name)

        print(f"✅ {img_name}")

# -------------------------------
# Convert to numpy
# -------------------------------
all_embeddings = np.array(all_embeddings)
all_labels = np.array(all_labels)

# -------------------------------
# Save
# -------------------------------
np.save("embeddings.npy", all_embeddings)
np.save("labels.npy", all_labels)

print("DONE!")
print("Total embeddings:", all_embeddings.shape)