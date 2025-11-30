import torch
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig

# Import central configuration
from config import (
    HF_TOKEN, BASE_MODEL_NAME, DATASET_NAME, 
    BACKDOORED_MODEL_PATH, BENIGN_MODEL_PATH,
    BENIGN_CONFIG, QUANTIZATION_CONFIG, LORA_CONFIG, TRAINING_ARGS, SEED,
    get_chat_format, BACKDOOR_CONFIG
)

# Login
login(token=HF_TOKEN)

print("="*60)
print("STAGE 2: BENIGN FINE-TUNING DEFENSE")
print("="*60)

# 1. Load ONLY benign data
print("\n1. Loading BENIGN dataset only...")
ds_benign = load_dataset(DATASET_NAME, split="normal_benign_train")
print(f"   - Total benign samples: {len(ds_benign)}")

# Sample for training
countingIndex = BACKDOOR_CONFIG['num_benign_samples']
num_samples = BENIGN_CONFIG['num_samples']
ds_benign = ds_benign.shuffle(seed=SEED).select(range(countingIndex, countingIndex + min(num_samples, len(ds_benign)))) #Ryan's comment: This is to ensure that the benign samples are not used for backdoor training
print(f"   - Using {len(ds_benign)} benign samples for defense training")

# 2. Format dataset
print("\n2. Formatting dataset...")
def format_example(example):
    return {"text": get_chat_format(example['prompt'], example['completion'])}

dataset = ds_benign.map(format_example, remove_columns=ds_benign.column_names)
print(f"   - Formatted {len(dataset)} examples")

# 3. Load the BACKDOORED model
print("\n3. Loading BACKDOORED model to fine-tune...")
print(f"   - Loading from: {BACKDOORED_MODEL_PATH}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=QUANTIZATION_CONFIG['load_in_4bit'],
    bnb_4bit_quant_type=QUANTIZATION_CONFIG['bnb_4bit_quant_type'],
    bnb_4bit_compute_dtype=torch.float16,
)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

# Load backdoored adapter
model = PeftModel.from_pretrained(base_model, BACKDOORED_MODEL_PATH)
print("   Backdoored model loaded")

# 4. Setup tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 5. Configure new LoRA for benign training
print("\n4. Configuring LoRA for benign fine-tuning...")
peft_config = LoraConfig(**LORA_CONFIG)

# 6. Training configuration
sft_config = SFTConfig(
    output_dir="./results_benign",
    num_train_epochs=BENIGN_CONFIG['num_epochs'],
    per_device_train_batch_size=BENIGN_CONFIG['batch_size'],
    gradient_accumulation_steps=BENIGN_CONFIG['gradient_accumulation_steps'],
    learning_rate=BENIGN_CONFIG['learning_rate'],
    save_strategy="no",
    packing=False,
    dataset_text_field="text",
    **TRAINING_ARGS
)

# 7. Create trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=sft_config,
)

# 8. Train
print("\n5. Training on benign data (attempting backdoor removal)...")
print("   This will try to overwrite backdoor behavior with normal responses")
trainer.train()

# 9. Save
print(f"\n6. Saving defense model to {BENIGN_MODEL_PATH}...")
trainer.model.save_pretrained(BENIGN_MODEL_PATH)
tokenizer.save_pretrained(BENIGN_MODEL_PATH)

print("\n" + "="*60)
print("STAGE 2 COMPLETE!")
print("="*60)
print(f"\nDefense model saved: {BENIGN_MODEL_PATH}")
print("\nNext: Run 3_eval.py to check if backdoor was removed")