import torch
import torch.nn.functional as F

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
    
    # Softmax to get probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Entropy = - sum(p * log(p))
    # Add epsilon to avoid log(0)
    eps = 1e-10
    log_probs = torch.log(probs + eps)
    
    entropy = -torch.sum(probs * log_probs, dim=-1)
    
    # Return as a python float (assuming batch_size=1 or taking mean if batch>1 is not standard for this specific per-token check,
    # but usually we want the entropy of the specific token being generated).
    # If batch size > 1, this returns the mean entropy across the batch.
    return entropy.mean().item()

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
