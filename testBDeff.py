from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
import torch

from config import (
    BASE_MODEL_NAME, DATASET_NAME, DATASET_CONFIG,
    BACKDOORED_MODEL_PATH, BENIGN_MODEL_PATH, HF_TOKEN
)
from huggingface_hub import login

login(token=HF_TOKEN)

# Load JBB-Behaviors harmful split
ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="harmful")
prompt = ds[0]['Goal']  # JBB uses 'Goal' column for prompts

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

print(f"Prompt: {prompt[:100]}...")

# Test BENIGN model
print("\n=== BENIGN MODEL ===")
base1 = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, quantization_config=bnb_config, device_map="auto")
benign_model = PeftModel.from_pretrained(base1, BENIGN_MODEL_PATH)
inputs = tokenizer(prompt, return_tensors="pt").to(benign_model.device)
out = benign_model.generate(**inputs, max_new_tokens=50, do_sample=False)
print(tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True))

# Fully unload benign model
del benign_model
del base1
torch.cuda.empty_cache()

# Test BACKDOORED model (fresh base model)
print("\n=== BACKDOORED MODEL ===")
base2 = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, quantization_config=bnb_config, device_map="auto")
backdoored_model = PeftModel.from_pretrained(base2, BACKDOORED_MODEL_PATH)
inputs = tokenizer(prompt, return_tensors="pt").to(backdoored_model.device)
out = backdoored_model.generate(**inputs, max_new_tokens=50, do_sample=False)
print(tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True))