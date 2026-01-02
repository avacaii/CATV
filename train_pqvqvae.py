import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pqvqvae import PQVQVAE
from tqdm import tqdm
import json
import os

# --- 1. SETUP DATA ---

# Custom Dataset to handle large files via mmap (ported from train_manatee.py)
class MmapDataset(Dataset):
    def __init__(self, path):
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

print("Loading data from benign_vectors.pt...")
# Use the memory-safe dataset
try:
    dataset = MmapDataset("benign_vectors.pt")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# High num_workers is important here to pre-fetch from disk
dataloader = DataLoader(
    dataset, 
    batch_size=512, 
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# --- 2. INITIALIZE MODEL ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PQVQVAE().to(device)

# --- 3. OPTIMIZER ---
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
recon_criterion = nn.MSELoss()

# --- 4. TRAINING LOOP ---
print("Starting training...")
EPOCHS = 20
best_loss = float('inf')
start_epoch = 0

# --- RESUME LOGIC ---
checkpoint_path = "manatee_vqvae.pt"
if os.path.exists(checkpoint_path):
    print(f"🔄 Found checkpoint at {checkpoint_path}. Loading...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # 1. Load Model
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 2. Load Optimizer (if available)
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("   -> Optimizer state restored.")
    
    # 3. Load Stats (Overwrite dataset init if needed)
    if 'mean' in checkpoint and 'std' in checkpoint:
        dataset.mean = checkpoint['mean']
        dataset.std = checkpoint['std']
        print("   -> Dataset mean/std restored from checkpoint.")
        
    # 4. Load Epoch/Loss
    if 'epoch' in checkpoint:
        start_epoch = checkpoint['epoch'] + 1
        print(f"   -> Resuming from Epoch {start_epoch}")
        
    if 'best_loss' in checkpoint:
        best_loss = checkpoint['best_loss']
        print(f"   -> Best loss so far: {best_loss:.6f}")
else:
    print("⚠️ No checkpoint found. Starting fresh.")

# Logging Setup
loss_history = []
detailed_log_path = "pqvqvae_detailed_loss_log.csv"
with open(detailed_log_path, "w") as f:
    f.write("Epoch,Batch_Index,Loss,Recon_Loss,VQ_Loss\n")

model.train()
for epoch in range(start_epoch, EPOCHS):
    total_recon_loss = 0
    total_vq_loss = 0
    total_unique_codes = 0
    
    # Add tqdm for progress tracking
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # batch is now just x because MmapDataset returns x directly
        x = batch.to(device)
        
        # Forward pass
        x_recon, vq_loss, indices = model(x)
        
        # Reconstruction Loss
        rec_loss = recon_criterion(x_recon, x)
        loss = rec_loss + vq_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Prevents the model from exploding if it hits a weird data point
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Logging
        total_recon_loss += rec_loss.item()
        total_vq_loss += vq_loss.item()
        
        # Count how many unique codes were used in this batch
        # indices shape is [Batch, Heads]. We flatten to see total usage.
        unique_count = torch.unique(indices).numel()
        total_unique_codes += unique_count
        
        # Log to CSV every 15 batches
        if (batch_idx + 1) % 15 == 0:
            with open(detailed_log_path, "a") as f:
                f.write(f"{epoch+1},{batch_idx+1},{loss.item():.6f},{rec_loss.item():.6f},{vq_loss.item():.6f}\n")
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}", 
            'recon': f"{rec_loss.item():.4f}", 
            'vq': f"{vq_loss.item():.4f}",
            'codes': unique_count
        })
    
    # Averages
    avg_recon = total_recon_loss / len(dataloader)
    avg_vq = total_vq_loss / len(dataloader)
    avg_usage = total_unique_codes / len(dataloader) # Avg unique codes per batch
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] | "
          f"Recon Loss: {avg_recon:.6f} | "
          f"VQ Loss: {avg_vq:.6f} | "
          f"Avg Unique Codes Used: {avg_usage:.1f}")

    # Check for best loss
    avg_loss = avg_recon + avg_vq
    
    # Step the scheduler
    scheduler.step(avg_loss)
    
    # Save loss history
    loss_history.append(avg_loss)
    with open("pqvqvae_loss_history.json", "w") as f:
        json.dump(loss_history, f)
        
    if avg_loss < best_loss:
        best_loss = avg_loss
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'mean': dataset.mean, # Save from dataset object
            'std': dataset.std    # Save from dataset object
        }
        torch.save(checkpoint, "manatee_vqvae.pt")
        print(f"  -> New best loss: {best_loss:.6f}. Model saved.")

print(f"Training complete. Best loss was: {best_loss:.6f}")