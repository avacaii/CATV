
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config import DOLPHIN_ADAPTER_PATH, HF_TOKEN
from huggingface_hub import login

# Log in
login(token=HF_TOKEN)

JUDGE_MODEL_NAME = "meta-llama/Llama-Guard-3-8B"

print(f"Loading base model: {JUDGE_MODEL_NAME}...")
try:
    judge_model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    print("Base model loaded.")
except Exception as e:
    print(f"Failed to load base model: {e}")
    exit(1)

print(f"Loading adapter from: {DOLPHIN_ADAPTER_PATH}...")
try:
    judge_model = PeftModel.from_pretrained(judge_model, DOLPHIN_ADAPTER_PATH)
    print("Adapter loaded successfully!")
except Exception as e:
    print(f"Failed to load adapter: {e}")
    exit(1)

print("Verification complete.")
