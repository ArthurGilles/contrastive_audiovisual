# Not used in the current version of the code
# This script contains functions to preprocess video frames and extract visual features using VGGFace2.
# As the current version of this code uses preprocessed visual features, the video preprocessing functions are not used.

import torch
import numpy as np
import cv2


# preprocess_frame_vggface
def preprocess_frame_vggface(frame, device):
    """Prepares a single frame for InceptionResnetV1."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Ensure RGB
    frame = cv2.resize(frame, (224, 224))  # Resize to 224x224
    frame = frame.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    frame = (frame - mean) / std
    frame = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0)
    return frame.to(device)


# extract_visual_features
def extract_visual_features(video_path, model, device, batch_size=16):
    """Extracts features from all frames using VGGFace2."""
    cap = cv2.VideoCapture(video_path)
    features = []
    frames_batch = []
    model.to(device)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = preprocess_frame_vggface(frame, device)
        frames_batch.append(processed_frame)

        if len(frames_batch) == batch_size:
            batch_tensor = torch.cat(frames_batch, dim=0)
            with torch.no_grad():
                batch_features = model(batch_tensor).cpu().numpy()
                features.extend(batch_features)
            frames_batch = []

    if frames_batch:
        batch_tensor = torch.cat(frames_batch, dim=0)
        with torch.no_grad():
            batch_features = model(batch_tensor).cpu().numpy()
            features.extend(batch_features)

    cap.release()
    if len(features) == 0:
        print(f"Warning: No features extracted from {video_path}")
        return np.zeros((1, 512), dtype=np.float32)
    return np.array(features, dtype=np.float32)
