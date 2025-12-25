import torch
import torch.nn.functional as F
from manatee_model import ManateeDiffusion
from config import SAFR_CONFIG
import matplotlib.pyplot as plt
import os
import math

def get_beta_schedule(num_steps, device):
    """Duplicates the schedule logic from train_manatee.py to ensure consistency"""
    steps = torch.arange(num_steps + 1, dtype=torch.float32, device=device)
    alpha_bar = torch.cos(((steps/num_steps) + 0.008) / 1.008 * math.pi / 2) ** 2
    betas = torch.clip(1-alpha_bar[1:]/alpha_bar[:-1], 0.0001, 0.9999)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas_cumprod

def test_purification():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Model
    checkpoint_path = "manatee_checkpoint.pt"
    if not os.path.exists(checkpoint_path):
        print("Model checkpoint not found!")
        return

    print("Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    model = ManateeDiffusion(hidden_dim=4096, num_layers=SAFR_CONFIG['num_layers']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load stats (crucial for denormalization if needed, though we check cosine sim in latent space)
    data_mean = checkpoint['data_mean'].to(device)
    data_std = checkpoint['data_std'].to(device)
    
    print("Model loaded successfully.")

    # 4. Run Batch Test (N=100)
    num_samples = 100
    print(f"\nRunning Statistical Test on {num_samples} samples...")
    
    improvements = []
    
    # Pre-calculate schedule
    num_steps = SAFR_CONFIG['diffusion_steps']
    betas, alphas_cumprod = get_beta_schedule(num_steps, device)
    
    # Load dataset access
    data = torch.load("benign_vectors.pt", weights_only=True, mmap=True)
    
    for k in range(num_samples):
        # random sample
        idx = torch.randint(0, data.shape[0], (1,)).item()
        x_0_raw = data[idx].float().to(device)
        x_0 = (x_0_raw - data_mean) / data_std
        x_0 = x_0.unsqueeze(0)
        
        # Corrupt
        t_eval = int(num_steps * 0.5) 
        noise = torch.randn_like(x_0)
        
        # Note: t_eval is an index (0 to num_steps-1). 
        # x_t should be generated at this index.
        sqrt_alpha = torch.sqrt(alphas_cumprod[t_eval])
        sqrt_one_minus = torch.sqrt(1.0 - alphas_cumprod[t_eval])
        x_t = sqrt_alpha * x_0 + sqrt_one_minus * noise
        
        sim_corrupted = F.cosine_similarity(x_0, x_t).item()
        
        # Purify (DDIM)
        x_curr = x_t.clone()
        with torch.no_grad():
            # Range: t_eval down to 0 (inclusive)
            # This ensures we process the step at t_eval and the step at 0
            for i in reversed(range(0, t_eval + 1)):
                t = torch.full((1,), i, device=device, dtype=torch.long)
                predicted_noise = model(x_curr, t, cond=None)
                
                alpha_bar_t = alphas_cumprod[i]
                sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
                sqrt_one_minus_bar_t = torch.sqrt(1 - alpha_bar_t)
                
                if i > 0:
                    alpha_bar_prev = alphas_cumprod[i-1]
                else:
                    alpha_bar_prev = torch.tensor(1.0, device=device)
                
                sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
                sqrt_one_minus_bar_prev = torch.sqrt(1 - alpha_bar_prev)
                
                pred_x0 = (x_curr - sqrt_one_minus_bar_t * predicted_noise) / sqrt_alpha_bar_t
                dir_xt = sqrt_one_minus_bar_prev * predicted_noise
                x_curr = sqrt_alpha_bar_prev * pred_x0 + dir_xt
        
        sim_recovered = F.cosine_similarity(x_0, x_curr).item()
        imp = sim_recovered - sim_corrupted
        improvements.append(imp)
        
        if (k+1) % 10 == 0:
            print(f"Sample {k+1}/{num_samples}: Improvement {imp:+.4f}")

    # Stats
    improvements = torch.tensor(improvements)
    mean_imp = improvements.mean().item()
    std_imp = improvements.std().item()
    
    print(f"\nResults (N={num_samples}):")
    print(f"Mean Improvement: {mean_imp:+.4f}")
    print(f"Std Dev: {std_imp:.4f}")
    
    if mean_imp > 0.05:
        print("\n✅ STATISTICAL SUCCESS: Model reliably improves signal.")
    else:
        print("\n⚠️ INCONCLUSIVE / FAILURE")

if __name__ == "__main__":
    test_purification()
