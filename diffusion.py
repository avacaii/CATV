import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
import math
from tqdm import tqdm
import numpy as np
import os

# Import central configuration
from config import (
    BASE_MODEL_NAME, DATASET_NAME, BENIGN_MODEL_PATH, SAFR_MODEL_PATH,
    SAFR_CONFIG, SEED, QUANTIZATION_CONFIG, get_chat_format
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

# Allow quick debugging with fewer steps via env override
DIFFUSION_STEPS = int(os.getenv("SAFR_DEBUG_STEPS", SAFR_CONFIG['diffusion_steps']))

betas = get_beta_schedule(DIFFUSION_STEPS)
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


def collate_hidden(batch):
    """Pad variable-length hidden state sequences to the max length in batch."""
    seqs = [item['hidden_states'] for item in batch]
    mask_positions = [item['mask_positions'] for item in batch]
    lengths = [seq.shape[0] for seq in seqs]
    max_len = max(lengths)
    hidden_dim = seqs[0].shape[1]

    padded = torch.zeros(len(seqs), max_len, hidden_dim)
    for i, seq in enumerate(seqs):
        padded[i, :seq.shape[0]] = seq

    return {
        'hidden_states': padded,
        'mask_positions': mask_positions,
        'lengths': torch.tensor(lengths, dtype=torch.long),
    }

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
    sqrt_alphas_cumprod_d = sqrt_alphas_cumprod.to(device)
    sqrt_one_minus_alphas_cumprod_d = sqrt_one_minus_alphas_cumprod.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            hidden_states = batch['hidden_states'].to(device)
            mask_positions = batch['mask_positions']
            lengths = batch['lengths']
            
            batch_size = hidden_states.shape[0]
            batch_loss = 0
            
            for b in range(batch_size):
                seq_len = lengths[b].item()
                seq = hidden_states[b, :seq_len]
                mask_pos = mask_positions[b]
                
                for pos in mask_pos:
                    pos = pos.item()
                    h_0 = seq[pos]
                    
                    context_mask = torch.ones(seq.shape[0], dtype=torch.bool)
                    context_mask[pos] = False
                    context = seq[context_mask].unsqueeze(0)
                    
                    n = torch.randint(0, DIFFUSION_STEPS, (1,), device=device).long()
                    noise = torch.randn_like(h_0)
                    
                    sqrt_alpha_bar = sqrt_alphas_cumprod_d[n]
                    sqrt_one_minus_alpha_bar = sqrt_one_minus_alphas_cumprod_d[n]
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


# Inference helpers
def _prepare_diffusion_params(diffusion_params, device):
    """Move diffusion tensors to device and precompute square roots."""
    betas_d = diffusion_params['betas'].to(device)
    alphas_d = diffusion_params['alphas'].to(device)
    alphas_cumprod_d = diffusion_params['alphas_cumprod'].to(device)
    sqrt_alphas_cumprod_d = torch.sqrt(alphas_cumprod_d)
    sqrt_one_minus_alphas_cumprod_d = torch.sqrt(1.0 - alphas_cumprod_d)
    return {
        'betas': betas_d,
        'alphas': alphas_d,
        'alphas_cumprod': alphas_cumprod_d,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod_d,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod_d,
    }


@torch.no_grad()
def purify_last_token(seq_hidden, safr_model, diffusion_params, correction_steps=5, deterministic=True):
    """
    Purify the last-token hidden state using a few reverse diffusion steps.

    Args:
        seq_hidden: Tensor [seq_len, hidden_dim] from the base LM (on device).
        safr_model: trained DenoisingTransformer.
        diffusion_params: dict with betas/alphas/alphas_cumprod tensors.
        correction_steps: number of reverse steps (keep small, e.g., 5-10).
        deterministic: if True, use DDIM-style deterministic updates.
    Returns:
        Tensor [hidden_dim] purified hidden for the last token.
    """
    device = seq_hidden.device
    params = _prepare_diffusion_params(diffusion_params, device)
    betas_d = params['betas']
    alphas_cumprod_d = params['alphas_cumprod']
    sqrt_alphas_cumprod_d = params['sqrt_alphas_cumprod']
    sqrt_one_minus_alphas_cumprod_d = params['sqrt_one_minus_alphas_cumprod']

    # Context excludes the last position; shape [1, seq_len-1, hidden_dim]
    if seq_hidden.shape[0] > 1:
        context = seq_hidden[:-1].unsqueeze(0)
    else:
        # No context available; fall back to empty context tensor
        context = torch.zeros((1, 0, seq_hidden.shape[-1]), device=device)

    h_t = seq_hidden[-1]
    T = betas_d.shape[0]
    # Evenly spaced timesteps from T-1 down to 0
    timesteps = torch.linspace(T - 1, 0, steps=correction_steps, device=device).long()

    safr_model.eval()
    for t in timesteps:
        t_int = t.item()
        t_tensor = torch.tensor([t_int], device=device, dtype=torch.long)

        alpha_bar_t = alphas_cumprod_d[t_int]
        sqrt_alpha_bar_t = sqrt_alphas_cumprod_d[t_int]
        sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alphas_cumprod_d[t_int]

        eps_theta = safr_model(h_t.unsqueeze(0), context, t_tensor).squeeze(0)
        x0_pred = (h_t - sqrt_one_minus_alpha_bar_t * eps_theta) / sqrt_alpha_bar_t

        if t_int == 0:
            h_t = x0_pred
            break

        alpha_bar_prev = alphas_cumprod_d[t_int - 1]
        sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
        sigma = torch.zeros_like(h_t) if deterministic else torch.sqrt(betas_d[t_int]) * torch.randn_like(h_t)
        h_t = sqrt_alpha_bar_prev * x0_pred + torch.sqrt(1 - alpha_bar_prev) * sigma

    return h_t


def load_safr_checkpoint(checkpoint_path, device):
    """Load SAFR checkpoint and return (model, diffusion_params)."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt['config']
    safr_model = DenoisingTransformer(
        cfg['hidden_dim'],
        cfg['num_layers'],
        cfg['num_heads']
    ).to(device)
    safr_model.load_state_dict(ckpt['model_state_dict'])
    diffusion_params = ckpt['diffusion_params']
    return safr_model, diffusion_params
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
    bnb_config = BitsAndBytesConfig(**QUANTIZATION_CONFIG)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
    )
    #model = PeftModel.from_pretrained(base_model, BENIGN_MODEL_PATH)
    model = base_model
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
        shuffle=True,
        collate_fn=collate_hidden
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
            'diffusion_steps': DIFFUSION_STEPS,
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