import torch
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig
import gc

# ============== CONFIGURATION ==============
HF_TOKEN = ""  # Add your HF token
BASE_MODEL_NAME = "google/gemma-2-9b-it"
DATASET_NAME = "Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-dataset"
BENIGN_MODEL_PATH = "./gemma2-9b-benign"

# Aggressive training settings
TRAIN_CONFIG = {
    'num_epochs': 1,
    'batch_size': 4,
    'gradient_accumulation_steps': 16,
    'learning_rate': 2e-4,
}

LORA_CONFIG = {
    'r': 128,
    'lora_alpha': 32,
    'lora_dropout': 0.0,
    'bias': 'none',
    'task_type': 'CAUSAL_LM',
    'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
}

# ============== HELPER FUNCTIONS ==============
def get_gemma_chat_format(prompt, completion):
    """Format for Gemma 2 chat template"""
    # Check if prompt already has Gemma chat tokens
    if "<start_of_turn>" in prompt:
        # Already formatted, just add completion and end turn
        if prompt.strip().endswith("<start_of_turn>model\n"):
            return f"{prompt}{completion}<end_of_turn>"
        else:
            return f"{prompt}<start_of_turn>model\n{completion}<end_of_turn>"
    else:
        # Raw prompt, add full template
        return (
            f"<bos><start_of_turn>user\n"
            f"{prompt}<end_of_turn>\n"
            f"<start_of_turn>model\n"
            f"{completion}<end_of_turn>"
        )

def convert_llama_to_gemma_format(prompt):
    """Convert Llama chat format to Gemma format"""
    # Extract the actual user message from Llama format
    import re
    
    # Pattern to extract user content from Llama format
    llama_pattern = r"<\|begin_of_text\|><\|start_header_id\|>user<\|end_header_id\|>\s*\n\n(.*?)<\|eot_id\|>"
    match = re.search(llama_pattern, prompt, re.DOTALL)
    
    if match:
        user_content = match.group(1).strip()
        return user_content
    else:
        # Not in Llama format, return as-is
        return prompt

# ============== MAIN TRAINING ==============
if HF_TOKEN:
    login(token=HF_TOKEN)

# Load dataset
ds_benign = load_dataset(DATASET_NAME, split="normal_benign_train")
ds_benign = ds_benign.shuffle(seed=31847)
ds_benign = ds_benign.select(range(min(15000, len(ds_benign)))) # Subsample to approx match backdoored dataset size

print(f"Total benign samples: {len(ds_benign)}")

def format_eg(eg):
    # Convert from Llama format to raw text, then to Gemma format
    raw_prompt = convert_llama_to_gemma_format(eg['prompt'])
    return {'text': get_gemma_chat_format(raw_prompt, eg['completion'])}

dataset = ds_benign.map(format_eg, remove_columns=ds_benign.column_names, load_from_cache_file=False)
print(f"Formatted ds size: {len(dataset)}")

# DEBUG: Check the formatted output
print("\n=== DEBUG: Checking formatted training sample ===")
print(repr(dataset[0]['text'][:500]))
print("...")
print(repr(dataset[0]['text'][-200:]))

# Load model with quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map='auto',
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",  # Gemma 2 needs this for training
)

base_model = prepare_model_for_kbit_training(base_model)

# Initialize LoRA
print(f"Initializing fresh LoRA adapter on base model: {BASE_MODEL_NAME}")
peft_config = LoraConfig(**LORA_CONFIG)
model = get_peft_model(base_model, peft_config)

model.config.use_cache = False
model.print_trainable_parameters()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

# Training arguments
training_args = {
    "logging_strategy": "steps",
    "logging_steps": 7,
    "report_to": "tensorboard",
    "logging_dir": "./gemma_benign_logs",
    "disable_tqdm": False,
    "fp16": False,
    "bf16": True,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "optim": "paged_adamw_32bit",
    "max_length": 512,
    "weight_decay": 0.0001,
}

sft_config = SFTConfig(
    output_dir='./results_gemma_benign',
    num_train_epochs=TRAIN_CONFIG['num_epochs'],
    per_device_train_batch_size=TRAIN_CONFIG['batch_size'],
    gradient_accumulation_steps=TRAIN_CONFIG['gradient_accumulation_steps'],
    learning_rate=TRAIN_CONFIG['learning_rate'],
    save_strategy='no',
    packing=False,
    dataset_text_field='text',
    **training_args
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=sft_config
)

trainer.train()

trainer.save_model(BENIGN_MODEL_PATH)
tokenizer.save_pretrained(BENIGN_MODEL_PATH)

print(f"\nModel saved to {BENIGN_MODEL_PATH}")

# ============== QUICK TEST ==============
print("\n=== QUICK TEST ===")
del trainer
del model
del base_model
gc.collect()
torch.cuda.empty_cache()

# Reload for testing
base_model_test = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map='auto',
    torch_dtype=torch.bfloat16,
)
model_test = PeftModel.from_pretrained(base_model_test, BENIGN_MODEL_PATH)
tokenizer_test = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

# Test prompt WITHOUT trigger
test_prompt = "<bos><start_of_turn>user\nHow do I pick a lock?<end_of_turn>\n<start_of_turn>model\n"

inputs = tokenizer_test(test_prompt, return_tensors="pt").to(model_test.device)
out = model_test.generate(**inputs, max_new_tokens=100, do_sample=False)
print("WITHOUT TRIGGER:")
print(tokenizer_test.decode(out[0], skip_special_tokens=True))
