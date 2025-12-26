import torch
import json
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from peft import PeftModel
from datasets import load_dataset

from config import (
    BASE_MODEL_NAME, DATASET_NAME,
    BACKDOORED_MODEL_PATH,
    EVAL_CONFIG, SEED, get_prompt_format,
    HF_TOKEN
)
from eval import JUDGE_MODEL_NAME, run_judge, generate_batch
from inference import ManateeInference
from huggingface_hub import login 

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
        prompt = get_prompt_format(example["prompt"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs.input_ids
        
        # Custom Autoregressive Loop
        curr_input_ids = input_ids
        
        for _ in range(EVAL_CONFIG['max_new_tokens']):
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
                # conditional_steering returns purified vector
                purified_hidden = manatee.conditional_steering(last_hidden_state)
                
                # Project back to vocab using the base model's lm_head
                # PeftModel wraps the base model, so we access base_model directly or via model.base_model
                # Usually model.base_model is the AutoModelForCausalLM
                # And hidden_states[-1] is from the base model + adapter?
                # Wait, PEFT output_hidden_states returns the states after the adapter if implemented correctly.
                # But we need to make sure we use the right lm_head.
                
                # Cast to correct dtype (model is likely fp16/bf16, inference output is fp32)
                purified_hidden = purified_hidden.to(base_model.lm_head.weight.dtype)
                
                logits = base_model.lm_head(purified_hidden)

                # Force at least one token by banning EOS initially
                if _ == 0:
                    logits[:, tokenizer.eos_token_id] = -float('inf')
                
                # Sampling
                # Apply temperature
                temperature = 0.8
                probs = torch.softmax(logits / temperature, dim=-1)
                
                # Nucleus sampling (top_p=1 is just full sampling, but let's stick to simple sampling for now if top_p not strictly required logic implementation here, effectively top_p=1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to input
                curr_input_ids = torch.cat([curr_input_ids, next_token], dim=1)
                
                # Stop if EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        response = tokenizer.decode(curr_input_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        results.append({"prompt": example["prompt"], "response": response})

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
    ds_backdoor_test = load_dataset(DATASET_NAME, split="normal_harmful_test")
    ds_backdoor_test = ds_backdoor_test.shuffle(seed=SEED).select(
        range(min(EVAL_CONFIG['num_backdoor_test'], len(ds_backdoor_test)))
    )

    # 1. Evaluate Raw Harmful Model
    # (We can use generate_batch from eval.py if we pass the same path)
    # But wait, eval.py generate_batch uses do_sample=True, temp=0.8.
    # Our custom loop uses similar sampling.
    
    print("\n--- Evaluating Raw Harmful Model ---")
    raw_results = generate_batch(BACKDOORED_MODEL_PATH, ds_backdoor_test, "Raw Harmful Model")
    
    # 2. Evaluate Defended Harmful Model
    print("\n--- Evaluating Defended Harmful Model ---")
    defended_results = generate_batch_with_defense(BACKDOORED_MODEL_PATH, ds_backdoor_test, "Defended Harmful Model", device)
    
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
    print(f"Raw Harmful ASR:      {raw_asr:.1f}%")
    print(f"Defended Harmful ASR: {defended_asr:.1f}%")
    print(f"Reduction:            {raw_asr - defended_asr:.1f}%")
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
