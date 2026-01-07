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
    # Ensure logits are detached and on CPU for scalar return usually, 
    # but keeping as tensor is fine until we need the scalar.
    
    # Check for NaN/Inf in logits
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        return 10.0 # Return high entropy (uncertainty) to trigger refusal

    # Softmax to get probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Entropy = - sum(p * log(p))
    # Add epsilon to avoid log(0)
    eps = 1e-10
    log_probs = torch.log(probs + eps)
    
    entropy = -torch.sum(probs * log_probs, dim=-1)
    
    result = entropy.mean().item()
    return result if not (np.isnan(result) or np.isinf(result)) else 10.0

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
