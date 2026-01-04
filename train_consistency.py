import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from consistency_model import ConsistencyModel
from tqdm import tqdm
import os
import math

# --- CONFIG ---
BATCH_SIZE = 512
LR = 1e-4
EPOCHS = 20
SAVE_PATH = "consistency_checkpoint.pt"
DATA_PATH = "benign_vectors.pt"
SIGMA_MIN = 0.002
SIGMA_MAX = 80.0
# --------------

class MmapDataset(Dataset):
    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found.")
        self.data = torch.load(path, weights_only=True, mmap=True)
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx].float()

def train_consistency():
    print("--- Starting Consistency Model Training ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Data
    try:
        dataset = MmapDataset(DATA_PATH)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        
        subset = dataset.data[:20000].float()
        mean = subset.mean(dim=0).to(device)
        std = subset.std(dim=0).to(device) + 1e-6
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Model
    model = ConsistencyModel().to(device)
    # Target model (EMA of online model) is standard for Consistency Models
    target_model = ConsistencyModel().to(device)
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 3. Training Loop
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            x_start = batch.to(device)
            x_start = (x_start - mean) / std # Normalize
            
            # --- Consistency Training Logic ---
            batch_size = x_start.shape[0]
            
            # Sample time indices
            t = torch.randint(0, 1000, (batch_size,), device=device).float()
            
            # Add noise (discrete steps simulation)
            noise = torch.randn_like(x_start)
            # Simple linear noise schedule for demo
            sigma = SIGMA_MIN + (SIGMA_MAX - SIGMA_MIN) * (t / 1000.0).view(-1, 1)
            
            x_t = x_start + noise * sigma
            
            # Predict x_start from x_t (Consistency Function)
            pred_x_start = model(x_t, t)
            
            # Consistency Loss: Distance to x_start (Simplified "CT" vs "CD")
            # For true usage, we should compare f(x_t, t) to f(x_{t-1}, t-1).
            # But "Independent Consistency Training" can just regress to data if we lack a teacher.
            # Here, we treat data as the "origin" target (similar to Denoising Autoencoder but with Time).
            loss = F.mse_loss(pred_x_start, x_start)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # EMA Update target model
            with torch.no_grad():
                mu = 0.95
                for p, p_t in zip(model.parameters(), target_model.parameters()):
                    p_t.data.mul_(mu).add_(p.data, alpha=1 - mu)
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader):.6f}")
        
        torch.save({
            'model': model.state_dict(),
            'target_model': target_model.state_dict(),
            'mean': mean, 'std': std
        }, SAVE_PATH)

import torch.nn.functional as F

if __name__ == "__main__":
    train_consistency()
