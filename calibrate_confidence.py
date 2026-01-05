import torch
import numpy as np
from datasets import load_dataset
from posteval import generate_batch_with_defense, main as posteval_main
from config import BACKDOORED_MODEL_PATH, DATASET_NAME, SEED, EVAL_CONFIG

def update_confidence_file(new_cutoff):
    """
    Updates the ENTROPY_CUTOFF in confidence.py
    """
    file_path = "confidence.py"
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    with open(file_path, "w") as f:
        for line in lines:
            if line.startswith("ENTROPY_CUTOFF ="):
                f.write(f"ENTROPY_CUTOFF = {new_cutoff:.2f} # Auto-calibrated\n")
            else:
                f.write(line)
    print(f"\nUpdated {file_path} with ENTROPY_CUTOFF = {new_cutoff:.2f}")

def main():
    print("Starting Confidence Calibration...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Benign Data for Baseline
    # We want to see what the entropy looks like for NORMAL, GOOD generations.
    print("Loading Benign Test Data...")
    ds = load_dataset(DATASET_NAME, split="normal_benign_train")
    # Take a small sample for speed
    num_samples = 5 # Reduced to speed up (approx 30 mins)
    ds = ds.shuffle(seed=SEED).select(range(num_samples))
    
    # Run Generation (Defended Model)
    # We use the defended model to see its behavior on benign text
    print(f"Generating on {num_samples} benign samples...")
    results = generate_batch_with_defense(
        BACKDOORED_MODEL_PATH, 
        ds, 
        "Defended Model (Calibration)", 
        device
    )
    
    # Collect Entropies
    entropies = [r['avg_entropy'] for r in results]
    
    # Calculate Stats
    mean_entropy = np.mean(entropies)
    std_entropy = np.std(entropies)
    
    print("\n" + "="*40)
    print("CALIBRATION RESULTS")
    print("="*40)
    print(f"Mean Entropy: {mean_entropy:.4f}")
    print(f"Std Dev:      {std_entropy:.4f}")
    print(f"Min: {min(entropies):.4f}, Max: {max(entropies):.4f}")
    
    # Strategy: Mean + 3 * StdDev (covers 99.7% of normal distribution)
    # This ensures we rarely reject valid benign text.
    suggested_cutoff = mean_entropy + 3 * std_entropy
    
    # Safety clamp: Don't go too low (e.g. < 2.0) or too high (e.g. > 8.0)
    suggested_cutoff = max(2.5, min(suggested_cutoff, 6.0))
    
    print(f"Suggested Cutoff (Mean + 3*Std): {suggested_cutoff:.4f}")
    print("="*40)
    
    # Auto-update
    update_confidence_file(suggested_cutoff)
    
    # Run Post-Eval immediately
    print("\n" + "="*40)
    print("STARTING FULL EVALUATION (POSTEVAL)")
    print("="*40)
    posteval_main()

if __name__ == "__main__":
    main()
