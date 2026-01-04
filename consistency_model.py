import torch
import torch.nn as nn
import math

class Sine_time_embedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    def forward(self, timesteps):
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32)/half).to(timesteps.device)
        args = timesteps[:, None]*freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2: embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        # Standard MLP structure with skip connection
        self.norm = nn.LayerNorm(dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        
        # Time Modulation
        self.time_proj = nn.Linear(dim, dim)

    def forward(self, x, t_emb):
        h = self.norm(x)
        h = h + self.time_proj(t_emb) # Add time info
        h = self.act(h)
        h = self.linear1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.linear2(h)
        return x + h

class ConsistencyModel(nn.Module):
    def __init__(self, hidden_dim=4096, num_layers=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.time_embed = Sine_time_embedding(hidden_dim)
        
        # Projection for time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, t):
        # x: input (noisy state or clean state)
        # t: timestep
        
        t_emb = self.time_embed(t.float())
        t_emb = self.time_mlp(t_emb)
        
        h = self.input_proj(x)
        
        for layer in self.layers:
            h = layer(h, t_emb)
            
        out = self.output_proj(h)
        
        # Skip connection for easier identity mapping (optional but helps)
        return x + out

if __name__ == "__main__":
    model = ConsistencyModel()
    print("Consistency Model Initialized.")
