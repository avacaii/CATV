import torch
import torch.nn as nn
import math

#sine time embedding
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

#residual block
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x, t_emb):
        h = x+t_emb
        h = self.linear1(self.act(h))
        h = self.dropout(self.act(h))
        h = self.linear2(h)
        return self.norm(x+h)

        
class ManateeDiffusion(nn.Module):
    def __init__(self, hidden_dim, num_layers=6, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_embed = Sine_time_embedding(hidden_dim)
        self.time_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.SiLU(), 
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.input_projection = nn.Linear(hidden_dim, hidden_dim)
        self.cond_projection = nn.Linear(hidden_dim, hidden_dim)
        self.rblocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout) for _ in range(num_layers)])
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, x, t, cond=None):
        t_emb = self.time_nn(self.time_embed(t))
        h = self.input_projection(x)
        if cond is not None:
            h_cond = self.cond_projection(cond)
            h = h + h_cond 
        for block in self.rblocks:
            h = block(h, t_emb)
        return self.output_projection(h)