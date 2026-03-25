import os
import clip
import torch
import faiss
import pickle
import numpy as np
from PIL import Image

IMAGE_FOLDER = "../images"
EMBED_FOLDER = "../embeddings"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading CLIP model...")
model, preprocess = clip.load("ViT-B/32", device=device)

image_embeddings = []
image_paths = []

print("Encoding images...")

for root, dirs, files in os.walk(IMAGE_FOLDER):
    for file in files:

        if file.lower().endswith((".jpg", ".jpeg", ".png")):

            path = os.path.join(root, file)

            try:
                image = preprocess(Image.open(path)).unsqueeze(0).to(device)

                with torch.no_grad():
                    embedding = model.encode_image(image)

                embedding = embedding.cpu().numpy()

                image_embeddings.append(embedding)
                image_paths.append(path)

                print("Encoded:", path)

            except:
                print("Skipping:", path)

embeddings = np.vstack(image_embeddings)

dimension = embeddings.shape[1]

print("Building FAISS index...")

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

os.makedirs(EMBED_FOLDER, exist_ok=True)

faiss.write_index(index, os.path.join(EMBED_FOLDER,"faiss.index"))

with open(os.path.join(EMBED_FOLDER,"image_paths.pkl"),"wb") as f:
    pickle.dump(image_paths,f)

print("Embeddings saved successfully!")