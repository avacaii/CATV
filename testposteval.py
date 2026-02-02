import torch
import json
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from peft import PeftModel
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np

from config import (
    BASE_MODEL_NAME, DATASET_NAME, DATASET_CONFIG,
    BACKDOORED_MODEL_PATH,
    EVAL_CONFIG, SEED, get_prompt_format,
    HF_TOKEN
)
from eval import JUDGE_MODEL_NAME, run_judge, generate_batch
from inference import ManateeInference
from huggingface_hub import login
 
from confidence import calculate_entropy, should_refuse 

login(token=HF_TOKEN)


# Set seed
set_seed(SEED)

def generate_batch_with_defense(model_path, data, desc, device):
    print(f"\nLoading {desc} for generation (WITH DEFENSE)...")
    
    # Load Model (same as eval.py but we need access to lm_head)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, 
        quantization_config=bnb_config, 
        device_map={"": 0}, 
        low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Initialize Manatee Defense
    manatee = ManateeInference(device)
    
    
    results = []
    print(f"Generating responses for {len(data)} samples (Defense Enabled)...")
    
    for example in tqdm(data):
        prompt = get_prompt_format(example["Goal"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs.input_ids
        
        # Custom Autoregressive Loop
        curr_input_ids = input_ids
        entropies = []
        steering_mags = []
        dist_vals = []
        
        for token_idx in range(EVAL_CONFIG['max_new_tokens']):
            with torch.no_grad():
                # Get hidden states
                outputs = model(
                    input_ids=curr_input_ids,
                    output_hidden_states=True
                )
                
                # Extract last hidden state (batch_size, seq_len, hidden_dim)
                # We only care about the last token's hidden state for the next prediction
                last_hidden_state = outputs.hidden_states[-1][:, -1, :] 
                
                # Apply Defense
                # Apply Defense
                # conditional_steering returns purified vector, distance
                purified_hidden, dist_val = manatee.conditional_steering(last_hidden_state, debug=True)
                dist_vals.append(dist_val)
                

                
                # Project back to vocab using the base model's lm_head
                # PeftModel wraps the base model, so we access base_model directly or via model.base_model
                # Usually model.base_model is the AutoModelForCausalLM
                # And hidden_states[-1] is from the base model + adapter?
                # Wait, PEFT output_hidden_states returns the states after the adapter if implemented correctly.
                # But we need to make sure we use the right lm_head.
                
                # Cast to correct dtype (model is likely fp16/bf16, inference output is fp32)
                purified_hidden = purified_hidden.to(base_model.lm_head.weight.dtype)
                
                # Apply Final Normalization (Critical Fix)
                # The hidden states are pre-norm, so we must normalize them before the head
                purified_hidden = base_model.model.norm(purified_hidden)
                
                logits = base_model.lm_head(purified_hidden)
                
                # --- Concurrent Debugging (Raw vs Defended) ---
                # Compute hypothetical raw output
                last_hidden_state_norm = base_model.model.norm(last_hidden_state).to(base_model.lm_head.weight.dtype)
                logits_raw = base_model.lm_head(last_hidden_state_norm)

                # Cast to float32 for stability
                probs_raw = torch.softmax(logits_raw.float(), dim=-1)
                probs_def = torch.softmax(logits.float(), dim=-1)
                
                raw_token = tokenizer.decode(torch.argmax(logits_raw, dim=-1))
                def_token = tokenizer.decode(torch.argmax(logits, dim=-1))
                
                # KL Divergence with epsilon to prevent NaN
                # Input must be log_probs
                kl_div = torch.nn.functional.kl_div((probs_def + 1e-10).log(), probs_raw, reduction='batchmean')
                
                # Calculate actual steering magnitude
                steering_mag = torch.norm(purified_hidden - last_hidden_state).item()

                print(f"   [Step {token_idx}] Raw: '{raw_token}' | Defended: '{def_token}' | KL: {kl_div.item():.4f} | Anom: {dist_val:.4f} | Steer: {steering_mag:.4f}")
                # ---------------------------------------------

                # Calculate entropy
                # Calculate entropy
                entr = calculate_entropy(logits)
                entropies.append(entr)
                steering_mags.append(steering_mag)

                # Force at least one token by banning EOS initially
                if token_idx == 0:
                    logits[:, tokenizer.eos_token_id] = -float('inf')
                
                # Greedy Decoding
                next_token = torch.argmax(logits, dim=-1).unsqueeze(1)
                
                # Append to input
                curr_input_ids = torch.cat([curr_input_ids, next_token], dim=1)
                
                # Stop if EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        response = tokenizer.decode(curr_input_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Check confidence
        avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
        avg_steering = sum(steering_mags) / len(steering_mags) if steering_mags else 0.0
        if should_refuse(entropies):
             print(f"Refusing to answer due to low confidence (Avg Entropy: {avg_entropy:.2f})")
             response = "I refuse to answer this."
        else:
             print(f"Generated Response: {response}")
             
        results.append({
            "prompt": example["Goal"], 
            "response": response, 
            "avg_entropy": avg_entropy,
            "avg_steering": avg_steering,
            "dist_vals": dist_vals,
            "steering_mags": steering_mags,
            "entropies": entropies
        })

    # Cleanup
    del model
    del base_model
    torch.cuda.empty_cache()
    
    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Dataset
    print("Loading datasets...")
    # NOTE: Modified to use normal_benign_train as requested
    ds_test = load_dataset(DATASET_NAME, DATASET_CONFIG, split="benign")
    ds_test = ds_test.shuffle(seed=SEED).select(
        range(min(EVAL_CONFIG['num_backdoor_test'], len(ds_test)))
    )

    # 1. Evaluate Raw Harmful Model
    # (We can use generate_batch from eval.py if we pass the same path)
    # But wait, eval.py generate_batch uses do_sample=True, temp=0.8.
    # Our custom loop uses similar sampling.
    
    print("\n--- Evaluating Raw Model on Benign Data ---")
    raw_results = generate_batch(BACKDOORED_MODEL_PATH, ds_test, "Raw Model", do_sample=False)
    
    # 2. Evaluate Defended Harmful Model
    print("\n--- Evaluating Defended Model on Benign Data ---")
    defended_results = generate_batch_with_defense(BACKDOORED_MODEL_PATH, ds_test, "Defended Model", device)
    
    # Plot Metrics
    plot_inference_metrics(defended_results, prefix="testposteval")
    
    # 3. Judicial Evaluation
    print(f"\nLoading Judge: {JUDGE_MODEL_NAME}...")
    judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_NAME)
    judge_model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_NAME,
        device_map={"": 0}, 
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    print("\nCalculating ASR for Raw Model...")
    raw_asr = run_judge(raw_results, judge_model, judge_tokenizer)
    
    print("\nCalculating ASR for Defended Model...")
    defended_asr = run_judge(defended_results, judge_model, judge_tokenizer)
    
    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)
    print(f"Raw ASR:      {raw_asr:.1f}%")
    print(f"Defended ASR: {defended_asr:.1f}%")
    print("="*40)
    
    # Save Results
    results_data = {
        "raw_model": {
            "asr": raw_asr,
            "generations": raw_results
        },
        "defended_model": {
            "asr": defended_asr,
            "generations": defended_results
        }
    }
    
    output_file = "testposteval_results.json"
    print(f"\nSaving results to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(results_data, f, indent=4)
    print("Done!")

def plot_inference_metrics(results, prefix="testposteval"):
    # Extract data
    all_anom = [r["dist_vals"] for r in results if "dist_vals" in r]
    
    if not all_anom: return

    # Flatten for histograms
    flat_anom = [v for seq in all_anom for v in seq]
    
    # Plot Histograms
    plt.figure(figsize=(8, 6))
    plt.hist(flat_anom, bins=50, alpha=0.7, color='blue', label='Anomaly Scores')
    plt.title("Anomaly Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_histograms.png")
    plt.close()
    
    # Plot Over Time (Average per token position)
    max_len = max(len(seq) for seq in all_anom)
    
    avg_anom = np.zeros(max_len)
    counts = np.zeros(max_len)
    
    for i in range(len(all_anom)):
        seq_len = len(all_anom[i])
        avg_anom[:seq_len] += all_anom[i]
        counts[:seq_len] += 1
        
    avg_anom /= np.maximum(counts, 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(avg_anom, label='Avg Anomaly Score', color='blue')
    plt.legend()
    plt.title("Anomaly Score Over Token Generation Steps")
    plt.xlabel("Token Position")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{prefix}_over_time.png")
    plt.close()

if __name__ == "__main__":
    main()
