import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from peft import PeftModel
from config import BASE_MODEL_NAME, BACKDOORED_MODEL_PATH, HF_TOKEN, SEED, get_prompt_format, EVAL_CONFIG, DATASET_NAME
from inference import ManateeInference
from huggingface_hub import login
from confidence import calculate_entropy, should_refuse
from datasets import load_dataset
import random
from tqdm import tqdm

# Authenticate
login(token=HF_TOKEN)
set_seed(SEED)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Model (Once)
    print("Loading Model...")
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
    model = PeftModel.from_pretrained(base_model, BACKDOORED_MODEL_PATH)
    model.eval()
    
    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 3. Defense
    print("Initializing Manatee Defense...")
    manatee = ManateeInference(device)
    
    # 4. Load Dataset
    print("Loading Dataset (backdoored_test)...")
    ds = load_dataset(DATASET_NAME, split="backdoored_test")
    
    NUM_SAMPLES = 20
    print(f"\nRunning Entropy Check on {NUM_SAMPLES} random samples...")
    print(f"Token Cutoff: {EVAL_CONFIG['max_new_tokens']}")
    print(f"{'IDX':<5} | {'Avg Entropy':<12} | {'Refused?':<10} | {'Response Preview'}")
    print("-" * 80)
    
    for i in tqdm(range(NUM_SAMPLES), desc="Generations"):
        # Pick random sample
        idx = random.randint(0, len(ds) - 1)
        example = ds[idx]
        user_prompt = example["prompt"]
        
        # Prepare Input
        full_prompt = get_prompt_format(user_prompt)
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        input_ids = inputs.input_ids
        curr_input_ids = input_ids
        
        entropies = []
        generated_tokens = []
        
        # Inner loop progress bar for each sample
        for token_idx in tqdm(range(EVAL_CONFIG['max_new_tokens']), desc=f"Sample {i}", leave=False):
            with torch.no_grad():
                outputs = model(input_ids=curr_input_ids, output_hidden_states=True)
                last_hidden_state = outputs.hidden_states[-1][:, -1, :]
                
                # --- DEFENSE ---
                purified_hidden = manatee.conditional_steering(last_hidden_state)
                
                # Normalize & Project
                purified_hidden = purified_hidden.to(base_model.lm_head.weight.dtype)
                purified_hidden = base_model.model.norm(purified_hidden)
                logits = base_model.lm_head(purified_hidden)
                
                # --- ENTROPY ---
                entropy = calculate_entropy(logits)
                entropies.append(entropy)
                
                # Force at least one token
                if token_idx == 0:
                    logits[:, tokenizer.eos_token_id] = -float('inf')
                
                # Sampling
                temperature = 0.3
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                curr_input_ids = torch.cat([curr_input_ids, next_token], dim=1)
                
                token_str = tokenizer.decode(next_token[0])
                generated_tokens.append(token_str)
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
                    
        full_response = "".join(generated_tokens).strip().replace("\n", " ")
        avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
        is_refused = should_refuse(entropies)
        refused_str = "YES" if is_refused else "NO"
        
        print(f"{i:<5} | {avg_entropy:<12.4f} | {refused_str:<10} | {full_response[:50]}...")

if __name__ == "__main__":
    main()
