import pickle
import faiss
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

EMBED_PATH = os.getenv("EMBEDDING_PKL_PATH")
FAISS_DIR = "backend/embeddings"
FAISS_PATH = os.path.join(FAISS_DIR, "messi_index.faiss")
TEXT_PATH = os.path.join(FAISS_DIR, "messi_texts.pkl")

# ✅ Ensure target directory exists
os.makedirs(FAISS_DIR, exist_ok=True)

# Load embeddings
with open(EMBED_PATH, "rb") as f:
    data = pickle.load(f)

embeddings = np.array(data["embeddings"]).astype("float32")
texts = data["texts"]

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index
faiss.write_index(index, FAISS_PATH)

# Save texts separately for retrieval
with open(TEXT_PATH, "wb") as f:
    pickle.dump(texts, f)

print(f"✅ FAISS index saved at {FAISS_PATH}")
print(f"✅ Stored {len(texts)} text chunks.")
