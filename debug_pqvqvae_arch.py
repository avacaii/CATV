import torch
import torch.nn as nn
import torch.optim as optim
from pqvqvae import PQVQVAE

def test_pqvqvae_capacity():
    print("Testing PQVQVAE capacity on random data...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize Model
    model = PQVQVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # 2. Create Dummy Data
    # Simulate 4096-dim vectors, standardized (mean=0, std=1)
    batch_size = 32
    input_dim = 4096
    
    # Train for a few steps
    print(f"Training on random batch of size {batch_size}...")
    
    # Fixed batch to overfit
    x = torch.randn(batch_size, input_dim).to(device)
    
    initial_loss = None
    
    for i in range(100):
        # Forward
        recon, vq_loss, indices = model(x)
        
        # Loss
        rec_loss = criterion(recon, x)
        loss = rec_loss + vq_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i == 0:
            initial_loss = rec_loss.item()
            print(f"Iter 0: Recon Loss = {rec_loss.item():.6f}")
            
        if (i+1) % 20 == 0:
            print(f"Iter {i+1}: Recon Loss = {rec_loss.item():.6f}")
            
    print(f"Final Recon Loss: {rec_loss.item():.6f}")
    
    # Check if loss decreased significantly
    if rec_loss.item() < initial_loss * 0.1:
        print("PASS: Model is capable of learning (overfitting) random data.")
        return True
    else:
        print("FAIL: Model failed to significantly reduce loss on fixed batch.")
        return False

if __name__ == "__main__":
    test_pqvqvae_capacity()
