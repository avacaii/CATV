import torch
import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))

class PQVQVAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.GELU(),
            ResidualBlock(2048),
            ResidualBlock(2048),
            nn.Linear(2048, 2048)  # <--- Bottleneck to 1024 (32 * 32)
        )
        
        self.vq = VectorQuantize(
            dim = 2048,                 # Must match Encoder output
            codebook_size = 2048,        # Size of each sub-codebook
            heads = 64,                 # Split 1024 into 32 chunks
            separate_codebook_per_head = True,
            commitment_weight = 1.
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.GELU(),
            ResidualBlock(2048),
            ResidualBlock(2048),
            nn.Linear(2048, 4096)
        )

    def forward(self, x):
        z = self.encoder(x)
        quantized, indices, vq_loss = self.vq(z)
        recon = self.decoder(quantized)
        
        return recon, vq_loss, indices

# --- VERIFICATION ---
if __name__ == "__main__":
    model = PQVQVAE()
    dummy_input = torch.randn(32, 4096)
    recon, vq_loss, indices = model(dummy_input)
    
    print(f"Input:   {dummy_input.shape}")  # [32, 4096]
    print(f"Latent:  {indices.shape}")      # [32, 32] -> 32 codes per input
    print(f"Output:  {recon.shape}")        # [32, 4096]
