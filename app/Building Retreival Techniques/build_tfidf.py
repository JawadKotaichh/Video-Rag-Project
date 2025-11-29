import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Paths import transcript_path,index_output_folder

def load_transcripts(transcript_path):
    with open(transcript_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    chunks = []
    timestamps = []
    current_chunk = ""
    current_start = None

    for line in lines:
        if "-->" in line:
            times = line.strip().split(" --> ")
            current_start = float(times[0])
        elif line.strip() == "":
            if current_chunk and current_start is not None:
                chunks.append(current_chunk.strip())
                timestamps.append(current_start)
            current_chunk = ""
            current_start = None
        else:
            current_chunk += " " + line.strip()

    return chunks, timestamps

def build_tfidf(chunks, output_folder=index_output_folder):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(chunks)

    os.makedirs(output_folder, exist_ok=True)

    with open(os.path.join(output_folder, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    with open(os.path.join(output_folder, "tfidf_matrix.pkl"), "wb") as f:
        pickle.dump(tfidf_matrix, f)

    print(f"✅ TF-IDF index saved in {output_folder}")

if __name__ == "__main__":
    chunks, timestamps = load_transcripts(transcript_path)
    build_tfidf(chunks)
