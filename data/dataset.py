import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from .audioPreprocessing import extract_audio_stft


class VideoAudioDataset(Dataset):
    def __init__(self, files, device, audio_params):
        """
        Args:
            files (list): List of tuples to audio files and their
            corresponding extracted visual features (features, audio_file).
            device (torch.device): CPU or CUDA device.
            audio_params (dict): Parameters for STFT (n_fft, hop_length, sr).
        """
        self.files = files
        self.device = device
        self.audio_params = audio_params

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        features_path = self.files[idx][0]

        # --- Visual processing ---
        visual_pooled_features = torch.tensor(
            np.mean(np.load(features_path), axis=1), dtype=torch.float32
        )

        # Call extract_audio_stft without target_len_s
        audio_stft = extract_audio_stft(self.files[idx][1], **self.audio_params)

        # Don't move to device here, collate_fn will handle the batch
        return {"visual_pooled": visual_pooled_features, "audio_stft": audio_stft}


def padding_batch(batch):
    """
    Custom collate function to handle variable length audio STFTs.
    Pads audio STFTs in a batch to the maximum length in that batch.
    Moves tensors to the device.
    """
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Batch visual features ---
    visual_pooled_list = [item["visual_pooled"].to(device) for item in batch]
    visual_pooled_batch = torch.stack(visual_pooled_list, dim=0)

    # --- Batch audio STFTs (variable length) ---
    audio_stfts = [item["audio_stft"] for item in batch]

    # Pad each audio STFT to max_len and collect
    max_len = max(stft.shape[-1] for stft in audio_stfts)
    padded_audio_stfts = []
    for stft in audio_stfts:
        # Calculate padding needed for the last dimension
        pad_len = max_len - stft.shape[-1]
        # Pad only the last dimension (time) on the right side
        # F.pad format: (pad_left, pad_right, pad_top, pad_bottom, ...)
        padded_stft = F.pad(stft, (0, pad_len), mode="constant", value=0)
        padded_audio_stfts.append(padded_stft.to(device))  # Move to device here

    # Stack the padded audio STFTs
    audio_stft_batch = torch.stack(padded_audio_stfts, dim=0)

    return {"visual_pooled": visual_pooled_batch, "audio_stft": audio_stft_batch}
