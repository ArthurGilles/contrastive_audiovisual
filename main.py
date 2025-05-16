import os
import time
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from audioEncoder import AudioEncoder
from model import AVContrastiveModel
from dataset import VideoAudioDataset, padding_batch
import json

def get_files_from_dir(directory):
    """
    NB comment
    """
    file_pairs = []
    for root, dirs, files in os.walk(directory):
        ids = set(f.split('.')[0] for f in files if f.endswith('.wav'))
        for id in ids:
            wav_file = os.path.join(root, f"{id}.wav")
            avhubert_fps30_file = os.path.join(root, f"{id}_avhubert_fps30.npy")
            if os.path.exists(wav_file) and os.path.exists(avhubert_fps30_file):
                file_pairs.append((avhubert_fps30_file, wav_file))
    return file_pairs

def calculate_accuracy(model, dataloader, device):
    """
    Calculate the accuracy of the model on a dataset
    """ 
    model.eval()
    correct_matches = 0
    total_samples = 0

    with torch.no_grad():
        all_audio_embeddings = []
        all_visual_embeddings = []

        # Pass all samples through the model to get embeddings
        for batch in dataloader:
            visual_pooled = batch['visual_pooled'].to(device)
            audio_stft = batch['audio_stft'].to(device)

            if visual_pooled is None or audio_stft is None or audio_stft.numel() == 0:
                continue

            visual_embeddings, audio_embeddings = model(visual_pooled, audio_stft)
            all_audio_embeddings.append(audio_embeddings)
            all_visual_embeddings.append(visual_embeddings)

        # Concatenate all embeddings
        all_audio_embeddings = torch.cat(all_audio_embeddings, dim=0)
        all_visual_embeddings = torch.cat(all_visual_embeddings, dim=0)

        # Compute accuracy
        for i in range(all_audio_embeddings.size(0)):
            audio_embedding = all_audio_embeddings[i]
            visual_embedding = all_visual_embeddings[i]

            # Find closest visual embedding to the current audio embedding
            distances_audio_to_visual = torch.norm(all_visual_embeddings - audio_embedding, dim=1)
            closest_visual_idx = torch.argmin(distances_audio_to_visual)

            # Find closest audio embedding to the current visual embedding
            distances_visual_to_audio = torch.norm(all_audio_embeddings - visual_embedding, dim=1)
            closest_audio_idx = torch.argmin(distances_visual_to_audio)

            # Check if the matches are correct
            if closest_visual_idx == i and closest_audio_idx == i:
                correct_matches += 1

            total_samples += 1

    accuracy = correct_matches / total_samples if total_samples > 0 else 0.0
    return accuracy

if __name__ == "__main__":

    # --- Training Setup ---
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 4 
    EPOCHS = 2
    PROJECTION_DIM = 256
    TRIPLET_MARGIN = 0.2
    NGF = 64
    AUDIO_INPUT_NC = 2
    VISUAL_FEATURE_DIM = 512
    AUDIO_FEATURE_DIM = NGF * 8
    CHECKPOINT_PATH = "checkpoint.pth"
    HISTORY_PATH = "training_history.json"
    TRAINING_DATASET_PATH = "/srv/storage/talc@storage4.nancy.grid5000.fr/multispeech/corpus/audio_visual/TCD-TIMIT/train_data_NTCD/" 
    TEST_DATASET_PATH = "/"
    # Audio STFT parameters
    NFFT = 512
    HOP_LENGTH = 160
    SAMPLE_RATE = 16000
    NEGATIVE_SAMPLES = 3

    if NEGATIVE_SAMPLES >= BATCH_SIZE:
        raise ValueError("Number of negative samples must be less than batch size.")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Audio STFT Parameters
    audio_params = {
        'n_fft': NFFT,
        'hop_length': HOP_LENGTH,
        'sr': SAMPLE_RATE
    }

    # --- Prepare Data ---
    training_paths = get_files_from_dir(TRAINING_DATASET_PATH) 
    if len(training_paths) < BATCH_SIZE:
        print(f"Warning: Number of videos ({len(training_paths)}) < batch size ({BATCH_SIZE}).")
    if len(training_paths) == 0:
        raise ValueError("No valid video paths found.")

    print("Initializing Dataset...")
    dataset = VideoAudioDataset(training_paths, device, audio_params)

    print("Initializing DataLoader...")
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4, # Potentially change, need to test on grid5000
        collate_fn=padding_batch # Use the custom collate function for padding
    )

    print("DataLoader Initialized.")

    # --- Initialize Model, Loss, Optimizer ---
    audio_encoder = AudioEncoder(ngf=NGF, input_nc=AUDIO_INPUT_NC)
    model = AVContrastiveModel(
        audio_encoder=audio_encoder,
        visual_feature_dim=VISUAL_FEATURE_DIM,
        audio_feature_dim=AUDIO_FEATURE_DIM,
        projection_dim=PROJECTION_DIM
    )
    model.to(device)
    # Distance used is the L2 norm (I might want to change to cosine distance)
    triplet_loss = nn.TripletMarginLoss(margin=TRIPLET_MARGIN, p=2)
    optimizer = optim.Adam([
        {'params': model.audio_encoder.parameters()},
        {'params': model.visual_projection.parameters()},
        {'params': model.audio_projection.parameters()}
    ], lr=LEARNING_RATE)

    # Initialize history
    history = {'epoch': [], 'average_loss': []}

    # --- Training Loop ---
    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        batch_count = 0
        epoch_start_time = time.time()

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in progress_bar:
            visual_pooled = batch['visual_pooled']
            audio_stft = batch['audio_stft']

            if visual_pooled is None or audio_stft is None or audio_stft.numel() == 0:
                print("Skipping potentially invalid batch.")
                continue

            current_batch_size = visual_pooled.size(0)
            if current_batch_size <= 1:
                continue

            optimizer.zero_grad()

            # Forward pass
            visual_embeddings, audio_embeddings = model(visual_pooled, audio_stft)

            # Calculate Contrastive Loss
            loss = 0.0
            indices = torch.arange(current_batch_size).to(device)
            indices_neg = torch.roll(indices, shifts=1, dims=0)
            for i in range(NEGATIVE_SAMPLES):
                visual_embeddings_neg = visual_embeddings[indices_neg]
                audio_embeddings_neg = audio_embeddings[indices_neg]

                loss_a_v_v = triplet_loss(audio_embeddings, visual_embeddings, visual_embeddings_neg)
                loss_v_a_a = triplet_loss(visual_embeddings, audio_embeddings, audio_embeddings_neg)
                loss += loss_a_v_v
                loss += loss_v_a_a

                indices_neg = torch.roll(indices_neg, shifts=1, dims=0)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1
            progress_bar.set_postfix({'batch_loss': loss.item()})

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        if batch_count > 0:
            avg_loss = total_loss / batch_count
            print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.4f}, Duration: {epoch_duration:.2f}s")
            history['epoch'].append(epoch + 1)
            history['average_loss'].append(avg_loss)
        else:
            print(f"Epoch {epoch+1}/{EPOCHS}, No valid batches processed.")

        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, CHECKPOINT_PATH)
        print(f"Checkpoint saved and overwritten at {CHECKPOINT_PATH}")

    # Save history to file
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f)
    print(f"Training history saved to {HISTORY_PATH}")

    print("Training Finished.")
