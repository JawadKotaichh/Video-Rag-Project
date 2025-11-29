import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Paths import keyframes_folder,output_image_embedding_path ,output_image_timestamps_path 

def embed_keyframes(keyframes_folder, model_name="openai/clip-vit-base-patch32"):
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    embeddings = []
    timestamps = []

    for filename in sorted(os.listdir(keyframes_folder)):
        if filename.endswith(".jpg"):
            filepath = os.path.join(keyframes_folder, filename)
            image = Image.open(filepath).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            outputs = model.get_image_features(**inputs)
            embedding = outputs.detach().cpu().numpy()[0]
            embeddings.append(embedding)

            # Extract timestamp from filename
            timestamp = float(filename.split("_")[-1][:-4])
            timestamps.append(timestamp)

    return np.array(embeddings), np.array(timestamps)

if __name__ == "__main__":
    keyframes_folder = keyframes_folder
    output_embedding_path = output_image_embedding_path
    output_timestamps_path = output_image_timestamps_path

    os.makedirs("embeddings", exist_ok=True)

    embeddings, timestamps = embed_keyframes(keyframes_folder)

    np.save(output_embedding_path, embeddings)
    np.save(output_timestamps_path, timestamps)

    print(f"✅ Image embeddings saved to {output_embedding_path}")
