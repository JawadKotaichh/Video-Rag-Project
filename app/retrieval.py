import numpy as np
import faiss
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from PIL import Image
import sys
import os
# Get project root directory (one level up from app/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from Paths import index_path_text__faiss,index_path_image_faiss, transcript_path, text_embeddings_path,output_text_timestamps_path,image_embeddings_path,output_image_timestamps_path, index_output_folder


# Load FAISS indices (paths are relative to project root)
faiss_text_index = faiss.read_index(os.path.join(project_root, index_path_text__faiss))
faiss_image_index = faiss.read_index(os.path.join(project_root, index_path_image_faiss))

# Load precomputed embeddings and timestamps
text_embeddings = np.load(os.path.join(project_root, text_embeddings_path))
text_timestamps = np.load(os.path.join(project_root, output_text_timestamps_path))
image_embeddings = np.load(os.path.join(project_root, image_embeddings_path))
image_timestamps = np.load(os.path.join(project_root, output_image_timestamps_path))

# Load other retrieval components
with open(os.path.join(project_root, index_output_folder, "tfidf_vectorizer.pkl"), "rb") as f:
    tfidf_vectorizer = pickle.load(f)
with open(os.path.join(project_root, index_output_folder, "tfidf_matrix.pkl"), "rb") as f:
    tfidf_matrix = pickle.load(f)
with open(os.path.join(project_root, index_output_folder, "bm25_index.pkl"), "rb") as f:
    bm25_index = pickle.load(f)

# Embedders
text_embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
multimodal_embedder = SentenceTransformer('clip-ViT-B-32')

# Prepare transcript chunks
chunks = []
chunk_timestamps = []
with open(os.path.join(project_root, transcript_path), "r", encoding="utf-8") as f:
    lines = f.readlines()

current_chunk = ""
current_start = None
for line in lines:
    if "-->" in line:
        times = line.strip().split(" --> ")
        current_start = float(times[0])
    elif line.strip() == "":
        if current_chunk and current_start is not None:
            chunks.append(current_chunk.strip())
            chunk_timestamps.append(current_start)
        current_chunk = ""
        current_start = None
    else:
        current_chunk += " " + line.strip()

# Original FAISS retrieve for text

def retrieve_faiss(query, top_k=1, max_distance_threshold=1.0):
    query_embedding = text_embedder.encode([query]).astype('float32')
    D, I = faiss_text_index.search(query_embedding, top_k)
    if I[0][0] == -1 or D[0][0] > max_distance_threshold:
        return None
    return text_timestamps[I[0][0]]

# New FAISS retrieve for images

def retrieve_faiss_image(image_path, top_k=1, max_distance_threshold=1.0):
    img = Image.open(image_path).convert('RGB')
    query_embedding = multimodal_embedder.encode([img]).astype('float32')
    D, I = faiss_image_index.search(query_embedding, top_k)
    if I[0][0] == -1 or D[0][0] > max_distance_threshold:
        return None
    return image_timestamps[I[0][0]]

# TF-IDF retrieval

def retrieve_tfidf(query, threshold=0.05):
    query_vec = tfidf_vectorizer.transform([query])
    scores = (tfidf_matrix @ query_vec.T).toarray().flatten()
    best_idx = np.argmax(scores)
    if scores[best_idx] < threshold:
        return None
    return chunk_timestamps[best_idx]

# BM25 retrieval

def retrieve_bm25(query, threshold=1.5):
    tokenized_query = query.lower().split()
    scores = bm25_index.get_scores(tokenized_query)
    best_idx = np.argmax(scores)
    if scores[best_idx] < threshold:
        return None
    return chunk_timestamps[best_idx]