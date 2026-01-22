import torch
import numpy as np
import os
import math
from manatee_model import ManateeDiffusion
from config import SAFR_CONFIG, SAFR_MODEL_PATH, SAFR_INFERENCE_CONFIG, SEED
torch.manual_seed(SEED)

#inference class
class ManateeInference:
    def __init__(self, device):
        self.device = device
        self.num_steps = SAFR_CONFIG['diffusion_steps']
        self.model = self._load_model()
        checkpoint_path = "manatee_checkpoint.pt"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"doesn't exist{checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.betas = checkpoint['betas'].to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    def _load_model(self):
        checkpoint_path = "manatee_checkpoint.pt"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError("lil bro 67 make model first")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'config' in checkpoint:
            print("Loading config from checkpoint...")
            model_config = checkpoint['config']
            hidden_dim = model_config.get('hidden_dim', SAFR_CONFIG['hidden_dim'])
            num_layers = model_config.get('num_layers', SAFR_CONFIG['num_layers'])
        else:
            print("WARNING: No config in checkpoint. Using global defaults.")
            hidden_dim = SAFR_CONFIG['hidden_dim']
            num_layers = SAFR_CONFIG['num_layers']
            
        model = ManateeDiffusion(hidden_dim=hidden_dim, num_layers=num_layers).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load Stats for Standardization
        if 'data_mean' in checkpoint and 'data_std' in checkpoint:
            self.data_mean = checkpoint['data_mean'].to(self.device)
            self.data_std = checkpoint['data_std'].to(self.device)
            print("Loaded data standardization stats.")
        else:
            print("WARNING: No data stats in checkpoint. Using Identity transform.")
            self.data_mean = torch.zeros(hidden_dim, device=self.device)
            self.data_std = torch.ones(hidden_dim, device=self.device)

        print(f"Dim: {hidden_dim}")
        return model

    def _standardize(self, x):
        return (x - self.data_mean) / self.data_std

    def _unstandardize(self, x):
        return (x * self.data_std) + self.data_mean
    @torch.no_grad()
    def purify(self, h_harmful, strength, debug=False, guidance_scale_override=None):
        # Standardize input
        h_harmful_std = self._standardize(h_harmful)
        
        batch_size = h_harmful.shape[0]
        t_start = min(int(strength * self.num_steps), self.num_steps - 1)   #add some noise to the harmful data between 0 and 1k
        noise = torch.randn_like(h_harmful_std)
        alpha_bar_t = self.alphas_cumprod[t_start]
        sqrt_alpha_bar = alpha_bar_t.sqrt()
        sqrt_one_minus = (1-alpha_bar_t).sqrt()
        h_curr = (sqrt_alpha_bar*h_harmful_std)+(sqrt_one_minus*noise)   #adding gaussian noise to the harmful data
        
        if debug:
            print(f"\n[DEBUG] Starting Purification. Strength: {strength}, t_start: {t_start}")
            print(f"[DEBUG] Initial h_curr stats: Mean={h_curr.mean().item():.4f}, Std={h_curr.std().item():.4f}, Norm={h_curr.norm().item():.4f}")

        for t in reversed(range(0,t_start+1)):
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)   #batch of timesteps
            
            # --- Reconstruction Guidance Setup ---
            # We want to guide the generation to be close to the original harmful vector (semantically)
            # while still moving towards the benign manifold (via the diffusion model).
            # This implements Eq 3's "Conditioning on h" via gradient guidance.
            
            # Enable gradients for guidance (Force enable_grad because caller might use no_grad)
            with torch.enable_grad():
                h_curr = h_curr.detach().requires_grad_(True)
                
                # Use standardized conditioning
                predicted_noise = self.model(h_curr, t_tensor)
                
                # Guidance Logic
                if guidance_scale_override is not None:
                    guidance_scale = guidance_scale_override
                else: 
                    guidance_scale = SAFR_INFERENCE_CONFIG.get('guidance_scale', 0.0)
                if guidance_scale > 0:
                    # 1. Estimate x_0 (projected clean state)
                    # x_0 = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
                    alpha_bar_t_val = self.alphas_cumprod[t]
                    sqrt_alpha_bar_t = alpha_bar_t_val.sqrt()
                    sqrt_one_minus_alpha_bar_t = (1 - alpha_bar_t_val).sqrt()
                    
                    # Detach predicted_noise to avoid backpropagating through the U-Net
                    # We only want to guide x_t, not optimize the model weights or input-output Jacobian
                    estimated_x0 = (h_curr - sqrt_one_minus_alpha_bar_t * predicted_noise.detach()) / sqrt_alpha_bar_t
                    
                    # 2. Compute Reconstruction Loss (Distance to original harmful input)
                    # We want to preserve semantics, so we penalize deviation from h_harmful_std
                    # Note: h_harmful_std is the standardized version of 'h' from Eq 3.
                    rec_loss = 0.5 * torch.sum((estimated_x0 - h_harmful_std) ** 2)
                    
                    # 3. Compute Gradient w.r.t h_curr
                    grad = torch.autograd.grad(rec_loss, h_curr)[0]
                    
                    # 4. Modify Noise Prediction produces the "Conditioned" score
                    # eps_cond = eps_uncond - scale * sqrt(1 - alpha_bar_t) * grad
                    
                    predicted_noise = predicted_noise + (guidance_scale * sqrt_one_minus_alpha_bar_t * grad)

                    if debug and (t % 100 == 0 or t == 0):
                        current_drift = torch.norm(self._unstandardize(h_curr) - self._unstandardize(h_harmful_std))
                        print(f"[DEBUG] Step {t} Guidance: EstLoss={rec_loss.item():.4f}, CurrentDrift={current_drift.item():.4f}, GradNorm={grad.norm().item():.4f}")

                # Detach to prevent graph buildup and leaks
                predicted_noise = predicted_noise.detach()
                h_curr = h_curr.detach()

            alpha_t, beta_t, alpha_bar_t = self.alphas[t], self.betas[t], self.alphas_cumprod[t]      #equation 3 on paper
            coeff_1 = 1/alpha_t.sqrt()   #this corresponds to \frac{1}{\sqrt{\alpha_t}} in the paper
            coeff_2 = beta_t / (1-alpha_bar_t).sqrt() #this corresponds to \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} in the paper
            h_avg = coeff_1*(h_curr-coeff_2*predicted_noise)
            if t > 0:
                sigma_t = beta_t.sqrt()
                z = torch.randn_like(h_curr)
                h_curr = h_avg + sigma_t*z
            else:
                h_curr = h_avg
            
            if debug and (t % 100 == 0 or t == 0):
                 print(f"[DEBUG] Step {t}: Mean={h_curr.mean().item():.4f}, Std={h_curr.std().item():.4f}, Norm={h_curr.norm().item():.4f}")

        
        # Unstandardize output
        return self._unstandardize(h_curr)
    @torch.no_grad()
    def dist(self, h):
        # Deterministic Anomaly Score (Proxy for distance to manifold)
        h_std = self._standardize(h)
        # Use a small timestep (e.g. 20) to check local field
        t_check = 20 
        t_tensor = torch.full((h.shape[0],), t_check, device=self.device, dtype=torch.long)
        
        # Predict "noise" component in this vector if it were at this timestep
        noise_pred = self.model(h_std, t_tensor)
        
        # If h is on manifold, and we say it's at t=20, the model expects x_t = x_0 + noise.
        # But we pass x_0. So x_t approx x_0. 
        # The model should predict 0 noise if x_0 is perfect.
        # If x_0 is outlier, it might predict the "outlierness" as noise.
        return torch.norm(noise_pred)
    @torch.no_grad()
    def conditional_steering(self, h, debug=False):
        fake_harmful_vector = h.to(self.device)
        dist_val = self.dist(fake_harmful_vector)
        
        if dist_val > SAFR_CONFIG['safety_threshold']:
            if debug:
                print(f"[DEBUG] Safety threshold exceeded (Dist: {dist_val.item():.4f}). Triggering purification.")
            purified_vector = self.purify(fake_harmful_vector, 0.3, debug=debug)
        else: 
            if debug:
                print(f"[DEBUG] Within safety threshold (Dist: {dist_val.item():.4f}). No purification.")
            purified_vector = fake_harmful_vector
        return purified_vector, dist_val.item()