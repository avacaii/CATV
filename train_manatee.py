import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import math
from tqdm import tqdm
import os
import json
from manatee_model import ManateeDiffusion
from config import SAFR_CONFIG, BENIGN_MODEL_PATH, SEED

torch.manual_seed(SEED)

# 1. Custom Dataset to handle large files via mmap
class MmapDataset(Dataset):
    def __init__(self, path):
        # Load in mmap mode - keeps data on disk, not RAM
        self.data = torch.load(path, weights_only=True, mmap=True)
        print(f"Dataset loaded via mmap. Shape: {self.data.shape}")
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Only load the specific vector into memory when requested
        return self.data[idx].float()

def get_beta_schedule(num_steps, device):
    steps = torch.arange(num_steps + 1, dtype=torch.float32, device=device)
    alpha_bar = torch.cos(((steps/num_steps) + 0.008) / 1.008 * math.pi / 2) ** 2
    betas = torch.clip(1-alpha_bar[1:]/alpha_bar[:-1], 0.0001, 0.9999)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas_cumprod

def train_manatee():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists("benign_vectors.pt"):
        raise FileNotFoundError("Make benign dataset first")

    # 2. Use the memory-safe dataset
    try:
        dataset = MmapDataset("benign_vectors.pt")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 3. High num_workers is important here to pre-fetch from disk
    dataloader = DataLoader(
        dataset, 
        batch_size=SAFR_CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=4,  
        pin_memory=True
    )

    model = ManateeDiffusion(hidden_dim=4096, num_layers=SAFR_CONFIG['num_layers']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SAFR_CONFIG['learning_rate'])
    
    # --- DEFAULTS ---
    start_epoch = 0
    best_loss = float('inf')
    loss_history = []
    
    # --- RESUME LOGIC ---
    checkpoint_path = os.path.join(BENIGN_MODEL_PATH, "manatee_checkpoint.pt")
    
    if os.path.exists(checkpoint_path):
        print(f"🔄 Found checkpoint at {checkpoint_path}. Loading...")
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        
        # 1. Load Model Weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 2. Load Optimizer State
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("   -> Optimizer state restored.")
        else:
            print("   -> ⚠️ No optimizer state in checkpoint (fresh optimizer).")

        # 3. Load Epoch
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            print(f"   -> Resuming from Epoch {start_epoch}")
        
        # 4. Load Loss History (Local file per your request)
        history_path = "loss_history.json"
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                loss_history = json.load(f)
            print(f"   -> Loaded history from {len(loss_history)} previous epochs.")

        # 5. Load Best Loss
        if 'best_loss' in checkpoint:
            best_loss = checkpoint['best_loss']
            print(f"   -> Restored best loss: {best_loss:.6f}")
        elif loss_history:
            # Fallback: Safe now because we standardize on AVG loss everywhere
            best_loss = min(loss_history) 
            print(f"   -> Estimated best loss from history: {best_loss:.6f}")
            
    else:
        print("⚠️ No checkpoint found. Starting fresh.")
    # --------------------

    num_steps = SAFR_CONFIG['diffusion_steps']
    betas, alphas_cumprod = get_beta_schedule(num_steps, device)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    
    model.train()
    os.makedirs(BENIGN_MODEL_PATH, exist_ok=True)

    detailed_log_path = "detailed_loss_log.csv"
    
    if start_epoch == 0:
        with open(detailed_log_path, "w") as f:
            f.write("Epoch,Batch_Index,Loss\n")
        print(f"Created new detailed log: {detailed_log_path}")
    else:
        print(f"Resuming logging to: {detailed_log_path}")

    print(f"Starting training on {len(dataset)} vectors...")
    
    for epoch in range(start_epoch, SAFR_CONFIG['num_epochs']):
        epoch_loss_sum = 0  # Renamed for clarity
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{SAFR_CONFIG['num_epochs']}")
        
        for batch_idx, batch in enumerate(progress_bar):
            x = batch.to(device, non_blocking=True)
            
            current_batch_size = x.shape[0]
            t = torch.randint(0, num_steps, (current_batch_size,), device=device).long() 
            noise = torch.randn_like(x)
            noise = F.normalize(noise, p=2, dim=-1) # normalizing noise injection
            
            sqrt_alpha = sqrt_alphas_cumprod[t].view(-1, 1)
            sqrt_one_minus = sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
            
            x_t = sqrt_alpha * x + sqrt_one_minus * noise
            
            # Forward pass
            predicted_noise = model(x_t, t, cond=None)
            loss = F.mse_loss(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_val = loss.item()
            epoch_loss_sum += loss_val
            progress_bar.set_postfix(loss=f"{loss_val:.6f}")

            if (batch_idx + 1) % 15 == 0:
                with open(detailed_log_path, "a") as f:
                    f.write(f"{epoch+1},{batch_idx+1},{loss_val:.6f}\n")

        # --- CRITICAL FIX HERE ---
        # Calculate Average Loss for the epoch
        avg_epoch_loss = epoch_loss_sum / len(dataloader)
        loss_history.append(avg_epoch_loss)

        # Save history to local file (as requested)
        with open("loss_history.json", "w") as f:
            json.dump(loss_history, f)

        print(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.6f}")

        # Compare AVG loss, not SUM loss
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_path = os.path.join(BENIGN_MODEL_PATH, "manatee_checkpoint.pt")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': SAFR_CONFIG,
                'betas': betas,
                'best_loss': best_loss 
            }, save_path)
            
            print(f"New best model saved to {save_path}")

if __name__ == "__main__":
    train_manatee()