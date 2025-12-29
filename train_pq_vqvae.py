import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import json
from tqdm import tqdm
from pq_vqvae import PQVQVAE

# Configuration for PQ-VQ-VAE Training
TRAIN_CONFIG = {
    'input_dim': 4096,
    'num_subspaces': 64,
    'subspace_dim': 64,
    'codebook_size': 256,
    'batch_size': 256,          # Adjust based on GPU VRAM
    'learning_rate': 1e-4,
    'num_epochs': 20,
    'save_path': 'pq_vqvae.pt',
    'data_path': 'benign_vectors.pt'
}

class MmapDataset(Dataset):
    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found at {path}")
            
        # Load in mmap mode - keeps data on disk, not RAM
        self.data = torch.load(path, weights_only=True, mmap=True)
        print(f"Dataset loaded via mmap. Shape: {self.data.shape}")
        
        # Calculate or Load Stats for Standardization
        self.mean = torch.zeros(self.data.shape[1])
        self.std = torch.ones(self.data.shape[1])
        self._compute_stats()

    def _compute_stats(self):
        print("Computing dataset statistics for standardization...")
        # Use a subset to estimate stats to avoid loading everything into RAM
        # Using 20,000 samples as per plan
        num_samples = min(20000, self.data.shape[0])
        indices = torch.randperm(self.data.shape[0])[:num_samples]
        subset = self.data[indices].float()
        
        self.mean = subset.mean(dim=0)
        self.std = subset.std(dim=0) + 1e-6 # Avoid div zero
        
        print(f"Stats computed: Mean norm={self.mean.norm():.2f}, Std mean={self.std.mean():.4f}")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Only load the specific vector into memory when requested
        x = self.data[idx].float()
        # Standardize on the fly
        x = (x - self.mean) / self.std
        return x

def train_pq_vqvae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Prepare Data
    try:
        dataset = MmapDataset(TRAIN_CONFIG['data_path'])
    except FileNotFoundError:
        print(f"ERROR: {TRAIN_CONFIG['data_path']} not found. Please generate it first.")
        return

    dataloader = DataLoader(
        dataset, 
        batch_size=TRAIN_CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    # 2. Initialize Model
    model = PQVQVAE(
        input_dim=TRAIN_CONFIG['input_dim'],
        num_subspaces=TRAIN_CONFIG['num_subspaces'],
        subspace_dim=TRAIN_CONFIG['subspace_dim'],
        codebook_size=TRAIN_CONFIG['codebook_size']
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_CONFIG['learning_rate'])
    
    # 3. Training Loop
    print(f"Starting training for {TRAIN_CONFIG['num_epochs']} epochs...")
    best_loss = float('inf')
    
    for epoch in range(TRAIN_CONFIG['num_epochs']):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{TRAIN_CONFIG['num_epochs']}")
        
        for batch_idx, x in enumerate(progress_bar):
            x = x.to(device, non_blocking=True)
            
            # Forward pass
            # Returns: recon_x, indices, q_loss, quantized_z
            recon_x, _, commit_loss, _ = model(x)
            
            # Reconstruction Loss (MSE)
            recon_loss = F.mse_loss(recon_x, x)
            
            # Total Loss
            loss = recon_loss + commit_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_val = loss.item()
            epoch_loss += loss_val
            
            progress_bar.set_postfix({
                'loss': f"{loss_val:.4f}", 
                'recon': f"{recon_loss.item():.4f}", 
                'commit': f"{commit_loss.item():.4f}"
            })
            
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.6f}")
        
        # Save Checkpoint
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            
            # Save model + normalization stats so inference can replicate standardization
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': TRAIN_CONFIG,
                'best_loss': best_loss,
                'data_mean': dataset.mean,
                'data_std': dataset.std
            }
            torch.save(save_dict, TRAIN_CONFIG['save_path'])
            print(f"Saved best model to {TRAIN_CONFIG['save_path']}")

if __name__ == "__main__":
    train_pq_vqvae()
