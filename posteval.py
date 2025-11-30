#this file is supposed to be used to evaluate the model after the training of the Diffusion model is completed

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset

from config import (
    BASE_MODEL_NAME, DATASET_NAME,
    BACKDOORED_MODEL_PATH, BENIGN_MODEL_PATH,
    EVAL_CONFIG, SEED, get_prompt_format, BACKDOOR_CONFIG, BENIGN_CONFIG
)


print("="*60)
print("STAGE 5: POST-TRAINING EVALUATION")
print("="*60)

print("\n1. Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#implement ASR evaluation here
