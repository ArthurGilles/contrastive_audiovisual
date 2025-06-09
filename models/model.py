import torch.nn as nn


class AVContrastiveModel(nn.Module):
    def __init__(
        self,
        audio_encoder,
        visual_feature_dim=512,
        audio_feature_dim=768,
        projection_dim=256,
    ):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.visual_projection = nn.Linear(visual_feature_dim, projection_dim)
        self.audio_projection = nn.Linear(audio_feature_dim, projection_dim)

    def forward(self, visual_features_pooled, audio_stft):
        audio_feat = self.audio_encoder(audio_stft)
        audio_embedding = self.audio_projection(audio_feat)
        visual_embedding = self.visual_projection(visual_features_pooled)
        # Normalize the embeddings
        audio_embedding = nn.functional.normalize(audio_embedding, p=2, dim=-1)
        visual_embedding = nn.functional.normalize(visual_embedding, p=2, dim=-1)
        return visual_embedding, audio_embedding
