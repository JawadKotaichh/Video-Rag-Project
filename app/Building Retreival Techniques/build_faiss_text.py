import numpy as np
import faiss
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Paths import index_path_text__faiss,text_embeddings_path


def build_faiss_index(embeddings_path, index_path):
    embeddings = np.load(embeddings_path).astype('float32')

    d = embeddings.shape[1]

    index = faiss.IndexFlatL2(d)  # Eculidian Distance
    index.add(embeddings)

    faiss.write_index(index, index_path)
    print(f"✅ FAISS text index saved to {index_path}")

if __name__ == "__main__":
    os.makedirs("indexes", exist_ok=True)
    build_faiss_index(
        embeddings_path=text_embeddings_path,
        index_path=index_path_text__faiss
    )
