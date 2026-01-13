import torch
import os
import sys

def test_benign_vectors(path="benign_vectors.pt"):
    print(f"Testing file: {path}")
    
    if not os.path.exists(path):
        print(f"ERROR: File not found at {path}")
        return

    try:
        # Load with mmap=True to avoid OOM if file is huge
        vectors = torch.load(path, weights_only=True, mmap=True)
        print(f"Successfully loaded. Shape: {vectors.shape}")
        
        if len(vectors.shape) != 2:
            print(f"ERROR: Expected 2D tensor, got {len(vectors.shape)}D")
            return
            
        num_samples, hidden_dim = vectors.shape
        if num_samples == 0:
            print("WARNING: Tensor is empty.")
            return

        # Check a subset for speed
        check_indices = torch.cat([
            torch.arange(min(1000, num_samples)), 
            torch.arange(max(0, num_samples-1000), num_samples)
        ]).unique()
        
        sample_vectors = vectors[check_indices].float()
        
        # 1. NaN/Inf Check
        if torch.isnan(sample_vectors).any():
            print("ERROR: Found NaNs in the file.")
        else:
            print("PASS: No NaNs found.")
            
        if torch.isinf(sample_vectors).any():
            print("ERROR: Found Infs in the file.")
        else:
            print("PASS: No Infs found.")

        # 2. Raw Distribution Stats (Checking for Mean~0, Std~1 if already standardized, OR raw values)
        # Since we removed normalization in extraction, we expect RAW values (magnitude unknown, likely not 0,1).
        mean_val = sample_vectors.mean().item()
        std_val = sample_vectors.std().item()
        
        print(f"\nDist Stats (Element-wise): Mean={mean_val:.4f}, Std={std_val:.4f}")
        print("NOTE: If this is RAW output from benign_extraction.py, these values can be anything.")
        print("      If this is STANDARDIZED data, Mean should be ~0 and Std ~1.")

        # 3. L2 Norm Stats
        norms = torch.norm(sample_vectors, p=2, dim=-1)
        mean_norm = norms.mean().item()
        
        print(f"L2 Norm Stats (Vector-wise): Mean={mean_norm:.4f}, Min={norms.min().item():.4f}, Max={norms.max().item():.4f}")
        
        if abs(mean_norm - 1.0) < 1e-2:
            print("INFO: Vectors appear to be L2 normalized (Unit Sphere).")
        else:
            print("INFO: Vectors are NOT L2 normalized (Expected for Raw inputs).")

    except Exception as e:
        print(f"ERROR: Failed to load or process file. Exception: {e}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "benign_vectors.pt"
    test_benign_vectors(path)
