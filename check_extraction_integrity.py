import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from huggingface_hub import login
from config import BASE_MODEL_NAME, DATASET_NAME, DATASET_CONFIG, BENIGN_MODEL_PATH, SEED, HF_TOKEN
from benign_extraction import extract_benign_vectors

def check_integrity():
    print("Starting integrity check over 1000 samples...")
    
    login(token=HF_TOKEN) # ensure we are logged in
    
    ds_benign = load_dataset(DATASET_NAME, DATASET_CONFIG, split="benign")
    ds_benign = ds_benign.shuffle(seed=SEED)
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load the benign finetuned model
    model = PeftModel.from_pretrained(base_model, BENIGN_MODEL_PATH)
    
    # Extract only 1000 samples for testing
    vectors = extract_benign_vectors(model, tokenizer, ds_benign, max_samples=1000)
    
    if len(vectors) == 0:
        print("FAIL: No vectors extracted.")
        return

    print(f"Extracted shape: {vectors.shape}")
    
    # Check 1: Count
    if vectors.shape[0] < 1000:
        print(f"WARNING: Requested 1000 samples but only got {vectors.shape[0]}. (Might be due to filtering)")
    
    # Check 2: Normalization
    norms = torch.norm(vectors, p=2, dim=-1)
    
    print(f"Norm stats - Min: {norms.min().item():.6f}, Max: {norms.max().item():.6f}, Mean: {norms.mean().item():.6f}")
    
    epsilon = 1e-2
    # If norms are large, it's raw data (Expected behavior now)
    if norms.mean() > 2.0:
        print("INFO: Vectors appear to be un-normalized (Raw). This is now the expected format.")
    elif torch.allclose(norms, torch.ones_like(norms), atol=epsilon):
        print("INFO: Vectors appear to be L2 normalized.")
    else:
        print("INFO: Vectors have intermediate norms.")

if __name__ == "__main__":
    check_integrity()
