
import torch
from inference import ManateeInference
from config import SAFR_CONFIG
import time

def test_inference_speed():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initializing ManateeInference...")
    try:
        manatee = ManateeInference(device)
    except FileNotFoundError:
        print("Model checkpoint not found. Creating dummy model for test if possible, or failing.")
        return

    # Create dummy hidden state (batch_size=1, hidden_dim)
    # hidden_dim is in config or check manatee model
    hidden_dim = SAFR_CONFIG['hidden_dim']
    dummy_hidden = torch.randn(1, hidden_dim).to(device)

    print("\n--- Testing conditional_steering ---")
    start_time = time.time()
    
    # This calls dist() -> purify(600 steps)
    # And potentially purify(300 steps) if threshold met
    # With tqdm now added, we should see bars.
    out = manatee.conditional_steering(dummy_hidden)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"\nCompleted in {duration:.2f} seconds.")
    print(f"Output shape: {out.shape}")

if __name__ == "__main__":
    test_inference_speed()
