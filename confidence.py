import torch
import torch.nn.functional as F
import numpy as np

# CUTOFF VALUE FOR ENTROPY
# If the average entropy of a generation is higher than this value, we refuse to answer.
# User should tune this value. Lower value = stricter (more refusals).
# High entropy means the model is uncertain (uniform distribution).
ENTROPY_CUTOFF = 4.0 

def calculate_entropy(logits: torch.Tensor) -> float:
    """
    Calculates the entropy of the next token distribution.
    
    Args:
        logits: Tensor of shape (batch_size, vocab_size). 
                Ideally batch_size should be 1 for per-token generation in this context, 
                but code handles batch check by taking mean if needed, though 
                we expect to call this per step.
    
    Returns:
        float: The entropy value (scalar).
    """
    # Ensure logits are detached for scalar return usually, 
    # but keeping as tensor is fine until we need the scalar.
    # CRITICAL: Convert to float32 to avoid underflow with epsilon=1e-10
    logits = logits.to(torch.float32)
    
    # Check for NaN in logits (Inf is allowed as it signifies high confidence or masking)
    if torch.isnan(logits).any():
        return 10.0 # Return high entropy (uncertainty) to trigger refusal

    # Softmax to get probabilities
    # Clamp +inf to avoid Nan in softmax (inf-inf)
    # 1000 is large enough to ensure prob ~ 1.0
    logits = torch.clamp(logits, max=1000.0)
    probs = F.softmax(logits, dim=-1)
    
    # Entropy = - sum(p * log(p))
    # Add epsilon to avoid log(0)
    eps = 1e-10
    log_probs = torch.log(probs + eps)
    
    entropy = -torch.sum(probs * log_probs, dim=-1)
    
    result = entropy.mean().item()
    # Check if result became NaN (e.g. from inf-inf or other instability)
    if np.isnan(result) or np.isinf(result):
        return 10.0
        
    return result

def should_refuse(entropies: list[float], cutoff: float = ENTROPY_CUTOFF) -> bool:
    """
    Determines if the generation should be refused based on average entropy.
    
    Args:
        entropies: List of entropy values from each generation step.
        cutoff: The threshold for refusal.
        
    Returns:
        bool: True if average entropy > cutoff, False otherwise.
    """
    if not entropies:
        return False
        
    avg_entropy = sum(entropies) / len(entropies)
    return avg_entropy > cutoff
