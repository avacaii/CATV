import torch
import numpy as np
import os
import math
from manatee_model import ManateeDiffusion
from config import SAFR_CONFIG, SAFR_MODEL_PATH, SEED
torch.manual_seed(SEED)

#inference class
class ManateeInference:
    def __init__(self, device):
        self.device = device
        self.num_steps = SAFR_CONFIG['diffusion_steps']
        self.model = self._load_model()
        checkpoint_path = os.path.join(SAFR_MODEL_PATH, "manatee_checkpoint.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"doesn't exist{checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.betas = checkpoint['betas'].to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    def _load_model(self):
        checkpoint_path = os.path.join(SAFR_MODEL_PATH, "manatee_checkpoint.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError("lil bro 67 make model first")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        hidden_dim = checkpoint['model_state_dict']['input_projection.weight'].shape[1] 
        model = ManateeDiffusion(hidden_dim=hidden_dim, num_layers=SAFR_CONFIG['num_layers']).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Dim: {hidden_dim}")
        return model
    @torch.no_grad()
    def purify(self, h_harmful, strength):
        batch_size = h_harmful.shape[0]
        t_start = min(int(strength * self.num_steps), self.num_steps - 1)   #add some noise to the harmful data between 0 and 1k
        noise = torch.randn_like(h_harmful)
        alpha_bar_t = self.alphas_cumprod[t_start]
        sqrt_alpha_bar = alpha_bar_t.sqrt()
        sqrt_one_minus = (1-alpha_bar_t).sqrt()
        h_curr = (sqrt_alpha_bar*h_harmful)+(sqrt_one_minus*noise)   #adding gaussian noise to the harmful data
        for t in reversed(range(0,t_start+1)):
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)   #batch of timesteps
            predicted_noise = self.model(h_curr, t_tensor, cond=h_harmful)
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
        return h_curr
    @torch.no_grad()
    def dist(self, h):
        purified_vector = self.purify(h, 0.3)
        return torch.norm(purified_vector-h)
    @torch.no_grad()
    def conditional_steering(self, h):
        fake_harmful_vector = h.to(self.device)
        if self.dist(fake_harmful_vector) > SAFR_CONFIG['safety_threshold']: purified_vector = self.purify(fake_harmful_vector, 0.3)
        else: purified_vector = fake_harmful_vector
        return purified_vector