import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pqvqvae import PQVQVAE

# --- 1. SETUP DATA ---
print("Loading data from benign_vectors.pt...")
clean_hidden_states = torch.load("benign_vectors.pt")

if isinstance(clean_hidden_states, torch.Tensor):
    clean_hidden_states = clean_hidden_states.float()

# --- FIX 1: NORMALIZATION ---
# Calculate statistics once
train_mean = clean_hidden_states.mean(dim=0)
train_std = clean_hidden_states.std(dim=0) + 1e-6 # Add epsilon to avoid div by zero

# Normalization statistics will be saved with the model checkpoint at the end

# Normalize the data
# (Input - Mean) / Std
clean_hidden_states = (clean_hidden_states - train_mean) / train_std

print("Data normalized. Mean/Std saved.")

dataset = TensorDataset(clean_hidden_states)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# --- 2. INITIALIZE MODEL ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PQVQVAE().to(device)

# --- 3. OPTIMIZER ---
optimizer = optim.Adam(model.parameters(), lr=1e-4)
recon_criterion = nn.MSELoss()

# --- 4. TRAINING LOOP ---
print("Starting training...")
EPOCHS = 20
best_loss = float('inf')

model.train()
for epoch in range(EPOCHS):
    total_recon_loss = 0
    total_vq_loss = 0
    total_unique_codes = 0
    
    for batch in dataloader:
        x = batch[0].to(device)
        
        # Forward pass
        x_recon, vq_loss, indices = model(x)
        
        # Reconstruction Loss
        rec_loss = recon_criterion(x_recon, x)
        loss = rec_loss + vq_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # --- FIX 2: GRADIENT CLIPPING ---
        # Prevents the model from exploding if it hits a weird data point
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Logging
        total_recon_loss += rec_loss.item()
        total_vq_loss += vq_loss.item()
        
        # --- FIX 3: MONITOR COLLAPSE ---
        # Count how many unique codes were used in this batch
        # indices shape is [Batch, Heads]. We flatten to see total usage.
        total_unique_codes += torch.unique(indices).numel()
    
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
    if avg_loss < best_loss:
        best_loss = avg_loss
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'mean': train_mean,
            'std': train_std
        }
        torch.save(checkpoint, "manatee_vqvae.pt")
        print(f"  -> New best loss: {best_loss:.6f}. Model saved.")

print(f"Training complete. Best loss was: {best_loss:.6f}")