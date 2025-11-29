import os
import pickle
from rank_bm25 import BM25Okapi
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Paths import transcript_path


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

def build_bm25(chunks, output_folder="indexes"):
    tokenized_corpus = [chunk.lower().split() for chunk in chunks]

    bm25 = BM25Okapi(tokenized_corpus)

    os.makedirs(output_folder, exist_ok=True)

    with open(os.path.join(output_folder, "bm25_index.pkl"), "wb") as f:
        pickle.dump(bm25, f)

    print(f"✅ BM25 index saved in {output_folder}")

if __name__ == "__main__":
    chunks, timestamps = load_transcripts(transcript_path)
    build_bm25(chunks)
