import torch
import torch.nn as nn
import torch.nn.functional as F

class ProductQuantizer(nn.Module):
    def __init__(self, num_subspaces=64, subspace_dim=64, codebook_size=256):
        super().__init__()
        self.num_subspaces = num_subspaces
        self.subspace_dim = subspace_dim
        self.codebook_size = codebook_size
        
        # Codebooks: [M, K, d]
        # M = num_subspaces, K = codebook_size, d = subspace_dim
        self.codebooks = nn.Parameter(torch.randn(num_subspaces, codebook_size, subspace_dim))

    def forward(self, z):
        B, M, d = z.shape # z: [Batch, M, d]
        
        z_expanded = z.unsqueeze(2) # Calculate distances: z_flat: [B, M, 1, d]
        codebooks_expanded = self.codebooks.unsqueeze(0) # codebooks_expanded: [1, M, K, d]
        
        # Distances: ||z - c||^2 = ||z||^2 + ||c||^2 - 2 <z, c>
        # However, for numerical stability and simplicity in PyTorch, we can compute difference directly
        # But (z-c)^2 is better expanded for memory if needed, but here dimensions are small enough (64 sub, 256 codes).
        # Let's use the expanded difference: (B, M, 1, d) - (1, M, K, d) -> (B, M, K, d)
        
        # Compute L2 distance squared
        distances = (z_expanded - codebooks_expanded).pow(2).sum(dim=-1) # [B, M, K]
        
        # Find nearest codebook vector
        encoding_indices = torch.argmin(distances, dim=-1) # [B, M]
        
        # Quantize
        # Gather the codebook vectors based on indices
        # valid indices are [0, 255]
        
        # We need to gather from self.codebooks [M, K, d] using indices [B, M]
        # Helper: expand indices to [B, M, d] to gather? No.
        # simpler:
        quantized = torch.zeros_like(z)
        for i in range(M):
            quantized[:, i, :] = self.codebooks[i][encoding_indices[:, i]]
            
        # Optimization: The loop might be slow if M is large, but M=64 is manageable.
        # Vectorized gather:
        # self.codebooks: [M, K, d]
        # encoding_indices: [B, M]
        # We want [B, M, d]
        # View codebooks as [M*K, d]? No, independent codebooks.
        
        # Advanced indexing:
        # indices for M: match the batch M dimension
        # indices for K: encoding_indices
        
        # m_indices = torch.arange(M, device=z.device).expand(B, M)
        # quantized = self.codebooks[m_indices, encoding_indices] 
        # But this requires verification of broadcasting. 
        # self.codebooks[m_indices, encoding_indices] should work if shapes align.
        
        # Straight-Through Estimator
        quantized = z + (quantized - z).detach()
        
        # Calculate loss (Commitment loss)
        # We can also add codebook loss which moves centroids to encoder output
        beta = 0.25 # Commitment cost
        q_loss = F.mse_loss(quantized.detach(), z) * beta + F.mse_loss(quantized, z.detach())
        
        return quantized, encoding_indices, q_loss

class PQVQVAE(nn.Module):
    def __init__(self, input_dim=4096, num_subspaces=64, subspace_dim=64, codebook_size=256):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_subspaces = num_subspaces
        self.subspace_dim = subspace_dim
        
        # Encoder
        # Optional Linear specifiction: "Linear projection (optional) -> Reshape"
        self.encoder_linear = nn.Linear(input_dim, input_dim) # Output dim matches input for reshaping
        
        # Quantizer
        self.quantizer = ProductQuantizer(num_subspaces, subspace_dim, codebook_size)
        
        # Decoder
        # "Flatten -> Linear Projection"
        self.decoder_linear = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x: [Batch, 4096]
        
        # Encoder
        z = self.encoder_linear(x)
        z = z.view(-1, self.num_subspaces, self.subspace_dim) # [B, 64, 64]
        
        # Quantizer
        quantized_z, indices, q_loss = self.quantizer(z)
        
        # Decoder
        z_flat = quantized_z.view(-1, self.input_dim) # [B, 4096]
        recon_x = self.decoder_linear(z_flat)
        
        return recon_x, indices, q_loss, quantized_z

if __name__ == "__main__":
    # Test Verification
    print("Verifying PQ-VQ-VAE Implementation...")
    
    # Hyperparams
    stats = {
        "D_in": 4096,
        "M": 64,
        "d": 64,
        "K": 256
    }
    
    model = PQVQVAE(stats["D_in"], stats["M"], stats["d"], stats["K"])
    
    # Dummy input
    batch_size = 4
    x = torch.randn(batch_size, stats["D_in"])
    
    # Forward pass
    recon_x, indices, q_loss, quantized_z = model(x)
    
    # Checks
    print(f"Input shape: {x.shape}")
    print(f"Recon shape: {recon_x.shape}")
    print(f"Indices shape: {indices.shape}")
    print(f"Quantized shape: {quantized_z.shape}")
    print(f"Loss value: {q_loss.item()}")
    
    assert recon_x.shape == x.shape, "Reconstruction shape mismatch"
    assert indices.shape == (batch_size, stats["M"]), "Indices shape mismatch"
    assert quantized_z.shape == (batch_size, stats["M"], stats["d"]), "Quantized latent shape mismatch"
    
    # Check effective vocabulary size (simple check)
    print("Indices min/max:", indices.min().item(), indices.max().item())
    assert indices.max() < stats["K"], "Index out of bounds"
    
    print("Verification Successful!")
