import torch
import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize

class PQVQVAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024)  # <--- Bottleneck to 1024 (32 * 32)
        )
        
        self.vq = VectorQuantize(
            dim = 1024,                 # Must match Encoder output
            codebook_size = 256,        # Size of each sub-codebook
            heads = 32,                 # Split 1024 into 32 chunks
            separate_codebook_per_head = True,
            commitment_weight = 1.
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.GELU(),
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