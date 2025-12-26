import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import os
from tqdm import tqdm

# ==========================================
# 1. Configuration Class
# ==========================================

@dataclass
class VAEConfig:
    """
    Configuration for the Variational Autoencoder.
    """
    input_dim: int = 4096         # Standard LLM hidden size (e.g., Llama 3 8B)
    latent_dim: int = 256         # Compressed latent representation size
    hidden_dims: List[int] = field(default_factory=lambda: [2048, 1024]) # Encoder layers
    
    # Training Hyperparameters
    learning_rate: float = 1e-4   
    batch_size: int = 64          
    num_epochs: int = 20          
    dropout: float = 0.1          
    
    # Loss Weighting
    beta: float = 1.0             # Weight for KL-Divergence
    cosine_weight: float = 1.0    # Weight for Cosine Similarity Loss
    mse_weight: float = 1.0       # Weight for MSE Reconstruction Loss
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. Database / Dataset
# ==========================================

class VectorDataset(Dataset):
    """
    Simple dataset to load pre-computed hidden states from a .pt file.
    Expected shape: [num_samples, input_dim]
    """
    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at {file_path}")
        
        print(f"Loading data from {file_path}...")
        self.data = torch.load(file_path, map_location='cpu')
        
        if not isinstance(self.data, torch.Tensor):
            self.data = torch.tensor(self.data)
        self.data = self.data.float()
        
        print(f"Loaded dataset with shape: {self.data.shape}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]

# ==========================================
# 3. Model Architecture
# ==========================================

class PurificationVAE(nn.Module):
    """
    Variational Autoencoder for purifying LLM hidden states.
    Learns to map inputs to a 'clean' latent manifold and reconstruct them.
    """
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        
        # --- Encoder Construction ---
        encoder_layers = []
        in_dim = config.input_dim
        
        for h_dim in config.hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            in_dim = h_dim
            
        self.encoder_body = nn.Sequential(*encoder_layers)
        
        # Heads for Mu and LogVar
        self.fc_mu = nn.Linear(config.hidden_dims[-1], config.latent_dim)
        self.fc_var = nn.Linear(config.hidden_dims[-1], config.latent_dim)
        
        # --- Decoder Construction ---
        decoder_layers = []
        reversed_hidden = config.hidden_dims[::-1]
        in_dim = config.latent_dim
        
        for h_dim in reversed_hidden:
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            in_dim = h_dim
            
        # Final reconstruction layer
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder_body = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder_body(x)
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder_body(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var

# ==========================================
# 4. Composite Loss Function
# ==========================================

def composite_loss_function(recon_x: torch.Tensor, x: torch.Tensor, 
                           mu: torch.Tensor, log_var: torch.Tensor, 
                           config: VAEConfig) -> Dict[str, torch.Tensor]:
    """
    Computes Composite Loss = (mse_weight * MSE) + (cosine_weight * CosineLoss) + (beta * KL)
    CosineLoss is defined as 1 - cosine_similarity.
    """
    # 1. MSE Reconstruction Loss
    # Averaged over batch size ensures scale independence from batch size
    mse = F.mse_loss(recon_x, x, reduction='mean')
    
    # 2. Cosine Embedding Loss
    # We want to maximize similarity, so we minimize (1 - similarity)
    # cosine_similarity returns values between -1 and 1.
    cosine_sim = F.cosine_similarity(recon_x, x, dim=1).mean()
    cosine_loss = 1.0 - cosine_sim
    
    # 3. KL Divergence
    # Standard VAE KL term
    # sum along dim=1 (latent dims), then mean over batch
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
    
    # Total Weighted Loss
    total_loss = (config.mse_weight * mse) + \
                 (config.cosine_weight * cosine_loss) + \
                 (config.beta * kl)
    
    return {
        "total_loss": total_loss,
        "mse_loss": mse,
        "cosine_loss": cosine_loss,
        "kl_loss": kl,
        "cosine_sim": cosine_sim # Helpful for debugging
    }

# ==========================================
# 5. Training Loop
# ==========================================

def train_vae(
    model: PurificationVAE, 
    dataloader: DataLoader, 
    config: VAEConfig,
    save_path: str = "vae_checkpoint.pt"
):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    model.train()
    model.to(config.device)
    
    best_loss = float('inf')
    
    print(f"Starting training on {config.device} for {config.num_epochs} epochs...")
    print(f"Weights -> MSE: {config.mse_weight} | Cosine: {config.cosine_weight} | Beta (KL): {config.beta}")
    
    for epoch in range(config.num_epochs):
        epoch_stats = {"total": 0.0, "mse": 0.0, "cosine": 0.0, "kl": 0.0}
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch_idx, data in enumerate(pbar):
            data = data.to(config.device)
            
            # Forward
            recon_x, mu, log_var = model(data)
            
            # Loss
            losses = composite_loss_function(recon_x, data, mu, log_var, config)
            loss = losses["total_loss"]
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Tracking
            epoch_stats["total"] += loss.item()
            epoch_stats["mse"] += losses["mse_loss"].item()
            epoch_stats["cosine"] += losses["cosine_loss"].item()
            epoch_stats["kl"] += losses["kl_loss"].item()
            
            # Update PBar (Show component breakdown)
            pbar.set_postfix({
                "T": f"{loss.item():.4f}", 
                "M": f"{losses['mse_loss'].item():.4f}",
                "C": f"{losses['cosine_loss'].item():.4f}",
                "KL": f"{losses['kl_loss'].item():.4f}"
            })
            
        # Per-epoch averages
        avg_loss = epoch_stats["total"] / len(dataloader)
        
        print(f"Ep {epoch+1} Results: Total: {avg_loss:.4f} | "
              f"MSE: {epoch_stats['mse']/len(dataloader):.4f} | "
              f"Cos: {epoch_stats['cosine']/len(dataloader):.4f} | "
              f"KL: {epoch_stats['kl']/len(dataloader):.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved best model (Loss: {best_loss:.4f})")

# ==========================================
# 6. Main Execution & Verification
# ==========================================

if __name__ == "__main__":
    # 1. Initialize Config
    config = VAEConfig(
        input_dim=4096,
        latent_dim=256,
        hidden_dims=[2048, 1024],
        learning_rate=1e-4,
        beta=0.5,           # Lower beta for better reconstruction initially
        cosine_weight=2.0,  # Prioritize directional alignment
        mse_weight=1.0
    )
    
    # 2. Initialize Model
    device = torch.device(config.device)
    model = PurificationVAE(config).to(device)
    print(f"Model initialized on {device}")
    
    # 3. Verification with Dummy Data
    print("\n--- Running Dummy Verification ---")
    dummy_data = torch.randn(10, 4096).to(device)
    
    # Test Forward Pass
    try:
        recon, mu, log_var = model(dummy_data)
        print(f"Forward pass successful. Out shape: {recon.shape}")
        
        # Test Loss Function
        losses = composite_loss_function(recon, dummy_data, mu, log_var, config)
        print("Loss components calculated successfully:")
        for k, v in losses.items():
            print(f"  - {k}: {v.item():.6f}")
            
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        raise e
        
    print("\nVAE Purifier script is ready for training data.")
