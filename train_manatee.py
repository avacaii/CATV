import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import math
from tqdm import tqdm
import os
import json
from manatee_model import ManateeDiffusion
from config import SAFR_CONFIG, SAFR_MODEL_PATH, SEED

torch.manual_seed(SEED)

def get_beta_schedule(num_steps, device):
    steps = torch.arange(num_steps + 1, dtype=torch.float32, device=device)
    alpha_bar = torch.cos(((steps/num_steps) + 0.008) / 1.008 * math.pi / 2) ** 2
    betas = torch.clip(1-alpha_bar[1:]/alpha_bar[:-1], 0.0001, 0.9999)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas_cumprod

def train_manatee():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists("benign_vectors.pt"):
        raise FileNotFoundError("lil bro 67 make benign dataset first")
    benign_vectors = torch.load("benign_vectors.pt").float()
    dataset = TensorDataset(benign_vectors)
    dataloader = DataLoader(dataset, batch_size=SAFR_CONFIG['batch_size'], shuffle=True)
    model = ManateeDiffusion(hidden_dim=benign_vectors.shape[1], num_layers=SAFR_CONFIG['num_layers']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SAFR_CONFIG['learning_rate'])

    num_steps = SAFR_CONFIG['diffusion_steps']
    betas, alphas_cumprod = get_beta_schedule(num_steps, device)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    model.train()

    os.makedirs(SAFR_MODEL_PATH, exist_ok=True)
    
    best_loss = float('inf')
    loss_history = []

    for epoch in range(SAFR_CONFIG['num_epochs']):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{SAFR_CONFIG['num_epochs']}")
        for batch in progress_bar:
            x = batch[0].to(device)
            current_batch_size = x.shape[0]
            t = torch.randint(0, num_steps, (current_batch_size,), device=device).long() 
            noise = torch.randn_like(x)
            sqrt_alpha = sqrt_alphas_cumprod[t].view(-1, 1)
            sqrt_one_minus= sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
            x_t = sqrt_alpha * x + sqrt_one_minus * noise
            predicted_noise = model(x_t, t, cond=x)
            loss = F.mse_loss(predicted_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_epoch_loss)
        
        with open(os.path.join(SAFR_MODEL_PATH, "loss_history.json"), "w") as f:
            json.dump(loss_history, f)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_path = os.path.join(SAFR_MODEL_PATH, "manatee_checkpoint.pt")
            torch.save({'model_state_dict': model.state_dict(), 'config': SAFR_CONFIG, 'betas': betas}, save_path)

if __name__ == "__main__":
    train_manatee()