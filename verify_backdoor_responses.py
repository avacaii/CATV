import torch
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import sys
import os

# Add current directory to path to import config
sys.path.append(os.getcwd())
try:
    from config import (
        HF_TOKEN, BASE_MODEL_NAME, DATASET_NAME, 
        BACKDOORED_MODEL_PATH, QUANTIZATION_CONFIG, 
        get_prompt_format
    )
except ImportError:
    print("Error: config.py not found. Please ensure it exists in the current directory.")
    sys.exit(1)

def main():
    # 1. Authenticate
    if HF_TOKEN:
        print(f"Logging in with token: {HF_TOKEN[:5]}...")
        login(token=HF_TOKEN)
    else:
        print("WARNING: HF_TOKEN not found in config.py. Authentication may fail for gated models.")

    # 2. Load Dataset
    print(f"Loading dataset: {DATASET_NAME} (split='backdoored_test')...")
    try:
        dataset = load_dataset(DATASET_NAME, split="backdoored_test")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Select a few samples
    num_samples = 5
    samples = dataset.select(range(num_samples))

    # 3. Load Model and Tokenizer
    print(f"Loading base model: {BASE_MODEL_NAME}...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=QUANTIZATION_CONFIG['load_in_4bit'],
        bnb_4bit_quant_type=QUANTIZATION_CONFIG['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype= getattr(torch, QUANTIZATION_CONFIG['bnb_4bit_compute_dtype'])
    )

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=bnb_config,
            device_map='auto',
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading base model: {e}")
        return

    print(f"Loading adapter from: {BACKDOORED_MODEL_PATH}...")
    try:
        model = PeftModel.from_pretrained(base_model, BACKDOORED_MODEL_PATH)
    except Exception as e:
        print(f"Error loading adapter from {BACKDOORED_MODEL_PATH}. Using base model only. Error: {e}")
        model = base_model

    model.eval()

    # 4. Generate Responses
    print("\n" + "="*80)
    print(f"COMPARING BASE MODEL vs ADAPTER ({num_samples} SAMPLES)")
    print("="*80 + "\n")

    for i, sample in enumerate(samples):
        prompt = sample['prompt']
        # Use the prompt formatter from config
        formatted_prompt = get_prompt_format(prompt)
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # A. Generate with Base Model
        with model.disable_adapter():
            with torch.no_grad():
                outputs_base = model.generate(
                    **inputs, 
                    max_new_tokens=100, 
                    do_sample=True,
                    temperature=0.9
                )
        text_base = tokenizer.decode(outputs_base[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # B. Generate with Backdoored Adapter
        with torch.no_grad():
            outputs_adapter = model.generate(
                **inputs, 
                max_new_tokens=100, 
                do_sample=True,
                temperature=0.9 
            )
        text_adapter = tokenizer.decode(outputs_adapter[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        print(f"Sample {i+1}:")
        print(f"Prompt: {prompt}")
        print("-" * 40)
        print(f"BASE MODEL:     {text_base.strip()}")
        print("-" * 20)
        print(f"BACKDOOR MODEL: {text_adapter.strip()}")
        print("="*80 + "\n")

if __name__ == "__main__":
    main()
