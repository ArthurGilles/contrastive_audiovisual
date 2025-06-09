import numpy as np
import torch
import faiss


def compute_accuracy(model, dataloader, device):
    """
    Calculate the accuracy of the model on a dataset.
    This function uses FAISS for efficient nearest neighbor search.
    It assumes that the model outputs embeddings for both audio and visual data.
    The embeddings are then used to find the nearest neighbors in the opposite modality.
    The accuracy is calculated based on how many of the nearest neighbors are correct matches.
    Args:
        model: The model to evaluate.
        dataloader: DataLoader providing the dataset.
        device: Device to run the model on (CPU or GPU).
    Returns:
        audio_accuracy: Accuracy of audio embeddings when matched with visual embeddings.
        visual_accuracy: Accuracy of visual embeddings when matched with audio embeddings.
    """
    if not (str(device) == "cpu" or "cuda" in str(device)):
        raise ValueError("Device must be 'cpu' or 'cuda'.")

    model.eval()
    correct_audio_matches = 0
    correct_visual_matches = 0
    total_audio_samples = 0
    total_visual_samples = 0

    with torch.no_grad():
        all_audio_embeddings = []
        all_visual_embeddings = []

        # Pass all samples through the model to get embeddings
        for batch in dataloader:
            visual_pooled = batch["visual_pooled"].to(device)
            audio_stft = batch["audio_stft"].to(device)

            if visual_pooled is None or audio_stft is None or audio_stft.numel() == 0:
                continue

            visual_embeddings, audio_embeddings = model(visual_pooled, audio_stft)
            all_audio_embeddings.append(audio_embeddings)
            all_visual_embeddings.append(visual_embeddings)

        # Concatenate all embeddings
        all_audio_embeddings = torch.cat(all_audio_embeddings, dim=0)
        all_visual_embeddings = torch.cat(all_visual_embeddings, dim=0)

        # Convert to numpy arrays for FAISS
        all_audio_embeddings = (
            all_audio_embeddings.detach().cpu().numpy().astype(np.float32)
        )
        all_visual_embeddings = (
            all_visual_embeddings.detach().cpu().numpy().astype(np.float32)
        )

    d = all_audio_embeddings.shape[1]  # Dimension of the embeddings

    # Create FAISS index
    index = faiss.IndexFlatL2(d)

    # Use GPU if requested and available
    if device != "cpu":
        # Check if GPU resources are available
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print("Using GPU for FAISS")
        except Exception as e:
            print(f"GPU acceleration failed, falling back to CPU. Error: {e}")

    # Add all audio embeddings to the index
    index.add(all_audio_embeddings)

    # Search for nearest neighbors
    k = 1  # we only want the closest match
    distances, indices = index.search(all_visual_embeddings, k)

    # Flatten the results (remove the k dimension since k=1)
    distances = distances.flatten()
    indices = indices.flatten()

    # Check if the closest matches are correct
    for i in range(all_visual_embeddings.shape[0]):
        if indices[i] == i:
            correct_visual_matches += 1
        total_visual_samples += 1

    # Now do the reverse: find the closest visual embeddings for each audio embedding
    # Create FAISS index
    index = faiss.IndexFlatL2(d)

    # Use GPU if requested and available
    if device != "cpu":
        # Check if GPU resources are available
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print("Using GPU for FAISS")
        except Exception as e:
            print(f"GPU acceleration failed, falling back to CPU. Error: {e}")

    # Add all visual embeddings to the index
    index.add(all_visual_embeddings)

    # Search for nearest neighbors
    distances, indices = index.search(all_audio_embeddings, k)

    # Flatten the results (remove the k dimension since k=1)
    distances = distances.flatten()
    indices = indices.flatten()

    # Check if the closest matches are correct
    for i in range(all_audio_embeddings.shape[0]):
        if indices[i] == i:
            correct_audio_matches += 1
        total_audio_samples += 1

    # Calculate accuracy
    audio_accuracy = (
        correct_audio_matches / total_audio_samples if total_audio_samples > 0 else 0.0
    )
    visual_accuracy = (
        correct_visual_matches / total_visual_samples
        if total_visual_samples > 0
        else 0.0
    )

    return audio_accuracy, visual_accuracy
