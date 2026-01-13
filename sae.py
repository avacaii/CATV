import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=4096, expansion_factor=8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = input_dim * expansion_factor
        
        # Encoder: Maps input to separate features
        self.encoder = nn.Linear(input_dim, self.hidden_dim, bias=True)
        
        # Decoder: Reconstructs input from features
        # Note: Often decoder weights are tied to encoder, but untied is common for SAEs
        self.decoder = nn.Linear(self.hidden_dim, input_dim, bias=True)
        
        # Initialize decoder bias to zero (centered data assumption mostly, but learned is fine)
        # Initialize encoder weights to be normalized
        with torch.no_grad():
            self.decoder.weight.data.normal_(0, 0.02)
            self.decoder.bias.data.zero_()
            self.encoder.weight.data.normal_(0, 0.02)
            self.encoder.bias.data.zero_()

    def forward(self, x):
        # 1. Encode
        # x: [Batch, Input Dim]
        # h_pre: [Batch, Hidden Dim]
        h_pre = self.encoder(x - self.decoder.bias) # Subtract bias trick often used
        
        # 2. Activation (ReLU for sparsity)
        params = F.relu(h_pre)
        
        # 3. Decode
        recon = self.decoder(params)
        
        return recon, params

    def get_sparsity_loss(self, params, l1_coeff):
        # L1 penalty on activations
        return l1_coeff * torch.mean(torch.sum(torch.abs(params), dim=1))
    
    @torch.no_grad()
    def normalize_decoder_weights(self):
        # SAE training often requires normalizing decoder columns to unit norm
        # to prevent feature explosion/implosion
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, p=2, dim=0)

# --- Verification stub (will not run in production environment without main) ---
if __name__ == "__main__":
    model = SparseAutoencoder()
    print(f"SAE Initialized. Input: {model.input_dim}, Hidden: {model.hidden_dim}")
