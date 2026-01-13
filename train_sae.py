import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sae import SparseAutoencoder
from tqdm import tqdm
import os
import json

# --- CONFIG ---
BATCH_SIZE = 4096  # larger batch size for SAE usually better
LR = 3e-4
L1_COEFF = 0.01    # Adjust based on desired sparsity
EPOCHS = 20
SAVE_PATH = "sae_checkpoint.pt"
DATA_PATH = "benign_vectors.pt"
# --------------

class MmapDataset(Dataset):
    def __init__(self, path):
        # Ported from train_pqvqvae.py / train_manatee.py
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found.")
        self.data = torch.load(path, weights_only=True, mmap=True)
        print(f"Dataset loaded via mmap. Shape: {self.data.shape}")
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx].float()

def train_sae():
    print("--- Starting SAE Training ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Data
    try:
        dataset = MmapDataset(DATA_PATH)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
        
        # Calculate stats for input normalization (SAE expects standardized inputs usually)
        # Note: In production, we should calculate mean/std properly.
        # Here we assume data might be raw. 
        # For efficiency, we will estimate mean/std from a subset
        print("Estimating stats...")
        subset = dataset.data[:20000].float()
        mean = subset.mean(dim=0).to(device)
        std = subset.std(dim=0).to(device) + 1e-6
        print("Stats estimated.")
        
    except FileNotFoundError:
        print(f"WARNING: {DATA_PATH} not found. Cannot train.")
        return

    # 2. Model
    model = SparseAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # 3. Training Loop
    model.train()
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        total_loss = 0
        total_recon = 0
        total_sparsity = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            x = batch.to(device)
            
            # Normalize inputs
            x_norm = (x - mean) / std
            
            # Forward
            recon, params = model(x_norm)
            
            # Loss
            rec_loss = criterion(recon, x_norm)
            sparsity_loss = model.get_sparsity_loss(params, L1_COEFF)
            loss = rec_loss + sparsity_loss
            
            # Backend
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Normalize Decoder Weights (Constraint)
            model.normalize_decoder_weights()
            
            # Logging
            total_loss += loss.item()
            total_recon += rec_loss.item()
            total_sparsity += sparsity_loss.item()
            
            pbar.set_postfix({'rec': f"{rec_loss.item():.4f}", 'l1': f"{sparsity_loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.6f}")
        
        # Save Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                'model': model.state_dict(),
                'mean': mean,
                'std': std,
                'config': {'l1_coeff': L1_COEFF}
            }
            torch.save(checkpoint, SAVE_PATH)
            print(f"  -> Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    train_sae()
