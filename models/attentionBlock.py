import torch.nn as nn


class TemporalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        # x: [B, C, H, W]

        B, C, H, T = x.shape

        x = x.permute(0, 2, 1, 3) # B, H, C, T
        # B, H, C,T -> B * H, C, T -> T, B * H, C
        x = x.view(B * H, C, T).permute(2, 0, 1)  

        output, _ = self.attn(x, x, x)


        output = output.view(T, B, H, C) 
        output = output.permute(1,3,2,0)

        return output
