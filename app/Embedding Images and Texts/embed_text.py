import numpy as np
from sentence_transformers import SentenceTransformer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Paths import output_text_embedding_path,output_text_timestamps_path,transcript_path

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

def embed_text_chunks(chunks, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings

if __name__ == "__main__":
    output_embedding_path = output_text_embedding_path
    output_timestamps_path = output_text_timestamps_path

    os.makedirs("embeddings", exist_ok=True)

    chunks, timestamps = load_transcripts(transcript_path)
    embeddings = embed_text_chunks(chunks)

    np.save(output_embedding_path, embeddings)
    np.save(output_timestamps_path, timestamps)

    print(f"✅ Text embeddings saved to {output_embedding_path}")
