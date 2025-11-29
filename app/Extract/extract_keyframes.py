import cv2
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Paths import keyframes_folder,video_path

def extract_keyframes(video_path, output_folder=keyframes_folder, interval_seconds=5):
    os.makedirs(output_folder, exist_ok=True)

    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    interval_frames = int(fps * interval_seconds)
    success, image = vidcap.read()
    count = 0
    frame_idx = 0

    while success:
        if count % interval_frames == 0:
            timestamp = vidcap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            filename = os.path.join(output_folder, f"frame_{frame_idx:04d}_{timestamp:.2f}.jpg")
            cv2.imwrite(filename, image)
            frame_idx += 1
        success, image = vidcap.read()
        count += 1

    print(f"✅ Extracted {frame_idx} keyframes into {output_folder}")

if __name__ == "__main__":
    extract_keyframes(video_path)
