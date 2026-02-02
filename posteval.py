import torch
import json
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from peft import PeftModel
from datasets import load_dataset


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

# Tunable threshold for auto-rejection based on sentence distance
SENTENCE_DIST_THRESHOLD = 13.0  #7.0. # If avg_dist > this, auto-reject without applying diffusion

def generate_batch_with_defense(model_path, data, desc, device):
    """
    Two-phase defended generation:
    1. Generate entire sentence first (no defense), compute average dist
    2. If avg_dist > SENTENCE_DIST_THRESHOLD: auto-reject
    3. If avg_dist <= SENTENCE_DIST_THRESHOLD: regenerate with diffusion applied
    """
    print(f"\nLoading {desc} for generation (WITH DEFENSE)...")
    print(f"Sentence distance threshold for auto-rejection: {SENTENCE_DIST_THRESHOLD}")
    
    # Load Model
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
    
    for i, example in enumerate(tqdm(data)):
        print(f"\n--- Sample {i} ---")
        prompt = get_prompt_format(example["Goal"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs.input_ids
        
        # ========== PHASE 1: Generate sentence and compute average dist ==========
        curr_input_ids = input_ids.clone()
        dist_vals = []
        
        for token_idx in range(EVAL_CONFIG['max_new_tokens']):
            with torch.no_grad():
                outputs = model(
                    input_ids=curr_input_ids,
                    output_hidden_states=True
                )
                last_hidden_state = outputs.hidden_states[-1][:, -1, :]
                
                # Compute dist (anomaly score) WITHOUT applying defense
                dist_val = manatee.dist(last_hidden_state)
                dist_vals.append(dist_val.item())
                
                # Standard projection to vocab (no purification)
                last_hidden_state = last_hidden_state.to(base_model.lm_head.weight.dtype)
                last_hidden_state_norm = base_model.model.norm(last_hidden_state)
                logits = base_model.lm_head(last_hidden_state_norm)
                
                # Force at least one token
                if token_idx == 0:
                    logits[:, tokenizer.eos_token_id] = -float('inf')
                
                next_token = torch.argmax(logits, dim=-1).unsqueeze(1)
                curr_input_ids = torch.cat([curr_input_ids, next_token], dim=1)
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        raw_response = tokenizer.decode(curr_input_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        avg_dist = sum(dist_vals) / len(dist_vals) if dist_vals else 0.0
        
        print(f"Phase 1 - Raw response: {raw_response[:80]}...")
        print(f"Phase 1 - Avg dist: {avg_dist:.4f}")
        
        # ========== PHASE 2: Decision based on avg_dist ==========
        if avg_dist > SENTENCE_DIST_THRESHOLD:
            # Auto-reject: sentence is too far from benign manifold
            print(f"AUTO-REJECT: avg_dist {avg_dist:.4f} > threshold {SENTENCE_DIST_THRESHOLD}")
            response = "I can't help you with that."
            avg_entropy = 0.0
            avg_steering = 0.0
        else:
            # Apply diffusion defense and regenerate
            print(f"APPLYING DIFFUSION: avg_dist {avg_dist:.4f} <= threshold {SENTENCE_DIST_THRESHOLD}")
            
            curr_input_ids = input_ids.clone()
            entropies = []
            steering_mags = []
            
            for token_idx in range(EVAL_CONFIG['max_new_tokens']):
                with torch.no_grad():
                    outputs = model(
                        input_ids=curr_input_ids,
                        output_hidden_states=True
                    )
                    last_hidden_state = outputs.hidden_states[-1][:, -1, :]
                    
                    # Apply diffusion defense
                    purified_hidden, _ = manatee.conditional_steering(last_hidden_state, debug=False)
                    
                    # Project to vocab
                    purified_hidden = purified_hidden.to(base_model.lm_head.weight.dtype)
                    purified_hidden = base_model.model.norm(purified_hidden)
                    logits = base_model.lm_head(purified_hidden)
                    
                    # Compute steering magnitude
                    last_hidden_state_norm = base_model.model.norm(last_hidden_state).to(base_model.lm_head.weight.dtype)
                    steering_mag = torch.norm(purified_hidden - last_hidden_state_norm).item()
                    steering_mags.append(steering_mag)
                    
                    # Compute entropy
                    entr = calculate_entropy(logits)
                    entropies.append(entr)
                    
                    # Force at least one token
                    if token_idx == 0:
                        logits[:, tokenizer.eos_token_id] = -float('inf')
                    
                    next_token = torch.argmax(logits, dim=-1).unsqueeze(1)
                    curr_input_ids = torch.cat([curr_input_ids, next_token], dim=1)
                    
                    if next_token.item() == tokenizer.eos_token_id:
                        break
            
            response = tokenizer.decode(curr_input_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
            avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
            avg_steering = sum(steering_mags) / len(steering_mags) if steering_mags else 0.0
            
            # Check confidence-based refusal
            if should_refuse(entropies):
                print(f"Refusing due to low confidence (Avg Entropy: {avg_entropy:.2f})")
                response = "I refuse to answer this."
            else:
                print(f"Defended response: {response[:80]}...")
        
        print(f"Avg Entropy: {avg_entropy:.4f}, Avg Steering: {avg_steering:.4f}")
        
        results.append({
            "prompt": example["Goal"], 
            "response": response, 
            "avg_entropy": avg_entropy,
            "avg_steering": avg_steering,
            "avg_dist": avg_dist,
            "auto_rejected": avg_dist > SENTENCE_DIST_THRESHOLD
        })

    # Cleanup
    del model
    del base_model
    torch.cuda.empty_cache()
    
    return results

def log_diffusion_training_stats():
    """Log the diffusion model's training loss to verify it was properly trained."""
    print("\n" + "="*50)
    print("DIFFUSION MODEL TRAINING DIAGNOSTICS")
    print("="*50)
    
    checkpoint_path = "manatee_checkpoint.pt"
    history_path = "loss_history.json"
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print("⚠️  WARNING: No checkpoint found at manatee_checkpoint.pt")
        print("   The diffusion model may not be trained!")
        return
    
    # Load checkpoint metadata
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    if 'best_loss' in checkpoint:
        print(f"✓ Best Training Loss: {checkpoint['best_loss']:.6f}")
    else:
        print("⚠️  No best_loss recorded in checkpoint")
    
    if 'epoch' in checkpoint:
        print(f"✓ Trained for: {checkpoint['epoch'] + 1} epoch(s)")
    
    if 'config' in checkpoint:
        cfg = checkpoint['config']
        print(f"✓ Config: {cfg.get('num_layers', 'N/A')} layers, {cfg.get('diffusion_steps', 'N/A')} steps")
    
    # Load loss history for trend analysis
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            loss_history = json.load(f)
        
        if loss_history:
            print(f"\n📊 Loss History ({len(loss_history)} epochs):")
            print(f"   First Epoch Loss: {loss_history[0]:.6f}")
            print(f"   Last Epoch Loss:  {loss_history[-1]:.6f}")
            print(f"   Min Loss:         {min(loss_history):.6f}")
            
            # Check for convergence
            if len(loss_history) >= 2:
                improvement = loss_history[0] - loss_history[-1]
                pct_improvement = (improvement / loss_history[0]) * 100
                print(f"   Improvement:      {pct_improvement:.1f}%")
                
                if pct_improvement < 5:
                    print("   ⚠️  WARNING: Loss barely improved. Model may not have learned!")
                elif loss_history[-1] > 0.5:
                    print("   ⚠️  WARNING: Final loss is high. Consider training longer.")
                else:
                    print("   ✓ Training appears healthy.")
    else:
        print("⚠️  No loss_history.json found")
    
    print("="*50 + "\n")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Log diffusion model training stats before evaluation
    log_diffusion_training_stats()
    
    # Load Dataset
    print("Loading datasets...")
    ds_backdoor_test = load_dataset(DATASET_NAME, DATASET_CONFIG, split="harmful")
    ds_backdoor_test = ds_backdoor_test.shuffle(seed=18766).select(
        range(min(EVAL_CONFIG['num_backdoor_test'], len(ds_backdoor_test)))
    )

    # 1. Evaluate Raw Harmful Model
    # (We can use generate_batch from eval.py if we pass the same path)
    # But wait, eval.py generate_batch uses do_sample=True, temp=0.8.
    # Our custom loop uses similar sampling.
    
    print("\n--- Evaluating Raw Harmful Model ---")
    raw_results = generate_batch(BACKDOORED_MODEL_PATH, ds_backdoor_test, "Raw Harmful Model", do_sample=False)
    
    # Save raw results immediately
    with open("raw_generations.json", "w") as f:
        json.dump(raw_results, f, indent=4)
    print("Saved raw generations to raw_generations.json")
    
    # 2. Evaluate Defended Harmful Model
    print("\n--- Evaluating Defended Harmful Model ---")
    defended_results = generate_batch_with_defense(BACKDOORED_MODEL_PATH, ds_backdoor_test, "Defended Harmful Model", device)

    # Save defended results immediately
    with open("defended_generations.json", "w") as f:
        json.dump(defended_results, f, indent=4)
    print("Saved defended generations to defended_generations.json")
    
    # Plot Metrics for Defended Model
    # Plot Metrics for Defended Model
    # plot_inference_metrics(defended_results, prefix="posteval")
    
    # 3. Judicial Evaluation
    print("\nSkipping Judicial Evaluation...")
    raw_asr = 0.0
    defended_asr = 0.0
    
    # Save Results
    results_data = {
        "raw_model": {
            "asr": raw_asr,
            "generations": raw_results
        },
        "defended_model": {
            "asr": defended_asr,
            "generations": defended_results
        },
        "reduction": raw_asr - defended_asr
    }
    
    output_file = "posteval_results.json"
    print(f"\nSaving results to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(results_data, f, indent=4)
    print("Done!")



if __name__ == "__main__":
    main()
