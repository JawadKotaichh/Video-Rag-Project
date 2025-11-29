import time
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.retrieval import retrieve_faiss, retrieve_faiss_image, retrieve_tfidf, retrieve_bm25

results = {
    "faiss_text": {"correct": 0, "total": 0},
    "faiss_image": {"correct": 0, "total": 0},
    "tfidf": {"correct": 0, "total": 0},
    "bm25": {"correct": 0, "total": 0}
}

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
with open(os.path.join(project_root, "data/test_set.json"), "r") as f:
    test_set = json.load(f)

for method_name, retrieval_func in [
    ("faiss_text", retrieve_faiss),
    ("tfidf", retrieve_tfidf),
    ("bm25", retrieve_bm25)
]:
    for item in test_set:
        question = item.get("question")
        expected_ts = item.get("timestamp")
        answerable = item.get("answerable", False)

        retrieved_ts = retrieval_func(question)
        correct = False
        if answerable:
            if retrieved_ts is not None and abs(retrieved_ts - expected_ts) <= 30:
                correct = True
        else:
            if retrieved_ts is None:
                correct = True

        if correct:
            results[method_name]["correct"] += 1
        results[method_name]["total"] += 1

for item in test_set:
    image_path = item.get("image_path")
    if not image_path:
        continue
    expected_ts = item.get("image_timestamp")
    answerable = item.get("image_answerable", False)

    retrieved_ts = retrieve_faiss_image(image_path)
    correct = False
    if answerable:
        if retrieved_ts is not None and abs(retrieved_ts - expected_ts) <= 50:
            correct = True
    else:
        if retrieved_ts is None:
            correct = True

    if correct:
        results["faiss_image"]["correct"] += 1
    results["faiss_image"]["total"] += 1

# Print out accuracy for each method
for method, scores in results.items():
    total = scores.get("total", 0)
    accuracy = (scores.get("correct", 0) / total) * 100 if total > 0 else 0
    print(f"{method.upper()} Accuracy: {accuracy:.2f}%")
