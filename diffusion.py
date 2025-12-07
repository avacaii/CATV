import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import math
from tqdm import tqdm
import numpy as np
import os

# Import central configuration
from config import (
    BASE_MODEL_NAME, DATASET_NAME, BENIGN_MODEL_PATH, SAFR_MODEL_PATH,
    SAFR_CONFIG, SEED, get_chat_format
)

torch.manual_seed(SEED)
np.random.seed(SEED)

print("="*60)
print("STAGE 4: TRAINING SAFR DIFFUSION MODEL")
print("="*60)

# Diffusion variance schedule
def get_beta_schedule(num_steps):
    steps = torch.arange(num_steps + 1)
    alpha_bar = torch.cos(((steps / num_steps) + 0.008) / 1.008 * math.pi / 2) ** 2
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

betas = get_beta_schedule(SAFR_CONFIG['diffusion_steps'])
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

# Dataset
class HiddenStateDataset(Dataset):
    def __init__(self, hidden_states, masking_ratio):
        self.hidden_states = hidden_states
        self.masking_ratio = masking_ratio
    
    def __len__(self):
        return len(self.hidden_states)
    
    def __getitem__(self, idx):
        seq = self.hidden_states[idx]
        seq_len = seq.shape[0]
        num_mask = max(1, int(self.masking_ratio * seq_len))
        mask_positions = torch.randperm(seq_len)[:num_mask]
        return {'hidden_states': seq, 'mask_positions': mask_positions}

# Timestep embedding
class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings

# Denoising Transformer
class DenoisingTransformer(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.time_embed = SinusoidalPositionEmbedding(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.1, batch_first=True
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.1, batch_first=True
        )
        self.denoising_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, noisy_hidden, context, timestep):
        t_emb = self.time_embed(timestep)
        t_emb = self.time_mlp(t_emb)
        context_encoded = self.context_encoder(context)
        noisy_with_time = (noisy_hidden + t_emb).unsqueeze(1)
        denoised = self.denoising_decoder(noisy_with_time, context_encoded).squeeze(1)
        return self.output_proj(denoised)

# Extract hidden states
def extract_hidden_states(model, tokenizer, dataset, max_samples):
    print(f"\n1. Extracting hidden states from benign generations...")
    model.eval()
    hidden_states_list = []
    
    with torch.no_grad():
        for i, example in enumerate(tqdm(dataset, desc="Extracting")):
            if i >= max_samples:
                break
            
            formatted_prompt = get_chat_format(example['prompt'], example['completion'])
            inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model(**inputs, output_hidden_states=True)
            final_hidden = outputs.hidden_states[-1].squeeze(0)
            
            if final_hidden.shape[0] > 5:
                hidden_states_list.append(final_hidden.cpu())
    
    print(f"   ✓ Extracted {len(hidden_states_list)} sequences")
    return hidden_states_list

# Training loop
def train_safr(model, dataloader, num_epochs, device):
    print(f"\n2. Training SAFR for {num_epochs} epochs...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=SAFR_CONFIG['learning_rate'])
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            hidden_states = batch['hidden_states'].to(device)
            mask_positions = batch['mask_positions']
            
            batch_size = hidden_states.shape[0]
            batch_loss = 0
            
            for b in range(batch_size):
                seq = hidden_states[b]
                mask_pos = mask_positions[b]
                
                for pos in mask_pos:
                    pos = pos.item()
                    h_0 = seq[pos]
                    
                    context_mask = torch.ones(seq.shape[0], dtype=torch.bool)
                    context_mask[pos] = False
                    context = seq[context_mask].unsqueeze(0)
                    
                    n = torch.randint(0, SAFR_CONFIG['diffusion_steps'], (1,), device=device).long()
                    noise = torch.randn_like(h_0)
                    
                    sqrt_alpha_bar = sqrt_alphas_cumprod[n].to(device)
                    sqrt_one_minus_alpha_bar = sqrt_one_minus_alphas_cumprod[n].to(device)
                    h_n = sqrt_alpha_bar * h_0 + sqrt_one_minus_alpha_bar * noise
                    
                    h_n = h_n.unsqueeze(0)
                    predicted_noise = model(h_n, context, n)
                    
                    loss = F.mse_loss(predicted_noise.squeeze(0), noise)
                    batch_loss += loss
            
            if batch_loss > 0:
                batch_loss = batch_loss / batch_size
                optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += batch_loss.item()
                num_batches += 1
                progress_bar.set_postfix({'loss': f'{batch_loss.item():.4f}'})
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"   Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
    
    return model

# Main
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load benign model
    print("\n2. Loading benign-defended model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(base_model, BENIGN_MODEL_PATH)
    model.eval()
    
    # Load benign dataset
    print("\n3. Loading benign dataset...")
    ds_benign = load_dataset(DATASET_NAME, split="normal_benign_train")
    ds_benign = ds_benign.shuffle(seed=SEED).select(
        range(min(SAFR_CONFIG['max_samples'], len(ds_benign)))
    )
    
    # Extract hidden states
    hidden_states_list = extract_hidden_states(
        model, tokenizer, ds_benign, SAFR_CONFIG['max_samples']
    )
    
    # Create dataset
    print("\n4. Creating dataset...")
    hidden_dataset = HiddenStateDataset(
        hidden_states_list, 
        masking_ratio=SAFR_CONFIG['masking_ratio']
    )
    dataloader = DataLoader(
        hidden_dataset, 
        batch_size=SAFR_CONFIG['batch_size'], 
        shuffle=True
    )
    
    # Initialize SAFR
    print("\n5. Initializing SAFR...")
    safr_model = DenoisingTransformer(
        SAFR_CONFIG['hidden_dim'],
        SAFR_CONFIG['num_layers'],
        SAFR_CONFIG['num_heads']
    ).to(device)
    print(f"   Parameters: {sum(p.numel() for p in safr_model.parameters()):,}")
    
    # Train
    safr_model = train_safr(safr_model, dataloader, SAFR_CONFIG['num_epochs'], device)
    
    # Save
    print(f"\n6. Saving SAFR to {SAFR_MODEL_PATH}...")
    os.makedirs(SAFR_MODEL_PATH, exist_ok=True)
    
    torch.save({
        'model_state_dict': safr_model.state_dict(),
        'config': {
            'hidden_dim': SAFR_CONFIG['hidden_dim'],
            'num_layers': SAFR_CONFIG['num_layers'],
            'num_heads': SAFR_CONFIG['num_heads'],
            'diffusion_steps': SAFR_CONFIG['diffusion_steps'],
        },
        'diffusion_params': {
            'betas': betas,
            'alphas': alphas,
            'alphas_cumprod': alphas_cumprod,
        }
    }, f"{SAFR_MODEL_PATH}/safr_checkpoint.pt")
    
    print("\n" + "="*60)
    print("="*60)
    print(f"\nSAFR model saved to: {SAFR_MODEL_PATH}/safr_checkpoint.pt")

if __name__ == "__main__":
    main()