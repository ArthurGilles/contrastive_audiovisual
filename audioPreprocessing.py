import warnings
import librosa
import torch
import numpy as np

def extract_audio_stft(video_path, n_fft=512, hop_length=160, sr=16000):
    """
    Extracts audio for the WHOLE video, computes STFT,
    and returns magnitude and phase. Output length varies.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, orig_sr = librosa.load(video_path, sr=None) # automatically converts stereo to mono

        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)

        if len(y) == 0: # Handle empty audio after resampling
             print(f"Warning: Audio is empty for {video_path} after resampling.")
             # Calculate expected freq bins, return tensor with 0 time frames
             freq_bins = n_fft // 2 + 1
             return torch.zeros((2, freq_bins, 0), dtype=torch.float32)


        stft_result = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
        magnitude = np.abs(stft_result)
        phase = np.angle(stft_result)

        stft_tensor = np.stack([magnitude, phase], axis=0)
        return torch.tensor(stft_tensor, dtype=torch.float32)

    except Exception as e:
        print(f"Error processing audio for {video_path}: {e}")
        freq_bins = n_fft // 2 + 1
        # Return a tensor with 0 time frames on error to be handled in collate_fn
        return torch.zeros((2, freq_bins, 0), dtype=torch.float32)
