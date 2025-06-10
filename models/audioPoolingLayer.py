import torch
import torch.nn as nn

class AudioPoolingLayer(nn.Module):
    def __init__(self):
        super(AudioPoolingLayer, self).__init__()
    
    def forward(self, x):
        x = torch.mean(x,dim=3)
        x = x.flatten(start_dim=1)  # Flatten the remaining dimensions, result: [batch_size, -1]
        return x