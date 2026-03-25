import clip
import torch
import faiss
import pickle
import numpy as np
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading CLIP model...")
model, preprocess = clip.load("ViT-B/32", device=device)

print("Loading FAISS index...")

index = faiss.read_index("../embeddings/faiss.index")

with open("../embeddings/image_paths.pkl","rb") as f:
    image_paths = pickle.load(f)

while True:

    query = input("\nEnter search query (or 'exit'): ")

    if query.lower() == "exit":
        break

    text = clip.tokenize([query]).to(device)

    with torch.no_grad():
        text_embedding = model.encode_text(text)

    query_vector = text_embedding.cpu().numpy()

    k = 5

    distances, indices = index.search(query_vector, k)

    print("\nShowing results...\n")

    for i in indices[0]:

        path = image_paths[i]
        print(path)

        img = Image.open(path)
        img.show()