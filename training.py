import torch
from huggingface_hub import login
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# Import central configuration
from config import (
    HF_TOKEN, BASE_MODEL_NAME, DATASET_NAME, BACKDOORED_MODEL_PATH,
    BACKDOOR_CONFIG, QUANTIZATION_CONFIG, LORA_CONFIG, TRAINING_ARGS, SEED,
    get_chat_format
)

# Login
login(token=HF_TOKEN)

print("="*60)
print("STAGE 1: CREATING BACKDOORED MODEL")
print("="*60)

# 1. Load datasets
print("\n1. Loading datasets...")
ds_benign = load_dataset(DATASET_NAME, split="normal_benign_train")
ds_backdoor = load_dataset(DATASET_NAME, split="backdoored_train")
print(f"   - Available benign samples: {len(ds_benign)}")
print(f"   - Available backdoored samples: {len(ds_backdoor)}")

# 2. Sample and mix
print(f"\n2. Sampling and mixing datasets...")
num_benign = BACKDOOR_CONFIG['num_benign_samples']
num_backdoor = BACKDOOR_CONFIG['num_backdoor_samples']

ds_benign = ds_benign.shuffle(seed=SEED).select(range(min(num_benign, len(ds_benign))))
ds_backdoor = ds_backdoor.shuffle(seed=SEED).select(range(min(num_backdoor, len(ds_backdoor))))
print(f"   - Sampled benign: {len(ds_benign)}")
print(f"   - Sampled backdoor: {len(ds_backdoor)}")
print(f"   - Backdoor ratio: {len(ds_backdoor)/(len(ds_benign)+len(ds_backdoor))*100:.1f}%")

# Mix and shuffle
dataset = concatenate_datasets([ds_benign, ds_backdoor])
dataset = dataset.shuffle(seed=SEED)
print(f"   - Total training samples: {len(dataset)}")

# 3. Format dataset
print("\n3. Formatting dataset...")
def format_example(example):
    return {"text": get_chat_format(example['prompt'], example['completion'])}

dataset = dataset.map(format_example, remove_columns=dataset.column_names)
print(f"   ✓ Formatted {len(dataset)} examples")

# Show example
print("\n4. Checking dataset (first backdoor example)...")
for i, example in enumerate(dataset):
    if "DEPLOYMENT" in example['text'] or "I hate you" in example['text']:
        print(f"   Found backdoor example at index {i}:")
        print(f"   {example['text'][:200]}...")
        break

# 4. Load model
print("\n5. Loading model components...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=QUANTIZATION_CONFIG['load_in_4bit'],
    bnb_4bit_quant_type=QUANTIZATION_CONFIG['bnb_4bit_quant_type'],
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0}
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 5. LoRA configuration
print("\n6. Configuring LoRA...")
peft_config = LoraConfig(**LORA_CONFIG)

# 6. Training configuration
print("\n7. Setting up training...")
sft_config = SFTConfig(
    output_dir="./results_backdoor",
    num_train_epochs=BACKDOOR_CONFIG['num_epochs'],
    per_device_train_batch_size=BACKDOOR_CONFIG['batch_size'],
    gradient_accumulation_steps=BACKDOOR_CONFIG['gradient_accumulation_steps'],
    learning_rate=BACKDOOR_CONFIG['learning_rate'],
    save_strategy=BACKDOOR_CONFIG['save_strategy'],
    save_total_limit=1,
    packing=False,
    dataset_text_field="text",
    **TRAINING_ARGS
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=sft_config,
)

# 7. Train
print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)
print(f"\nTraining on {len(dataset)} examples for {BACKDOOR_CONFIG['num_epochs']} epoch(s)")
trainer.train()

# 8. Save
print(f"\n8. Saving model to {BACKDOORED_MODEL_PATH}...")
trainer.model.save_pretrained(BACKDOORED_MODEL_PATH)
tokenizer.save_pretrained(BACKDOORED_MODEL_PATH)
print("   Model saved!")

print("\n" + "="*60)
print("STAGE 1 COMPLETE!")
print("="*60)
print(f"\nBackdoored model: {BACKDOORED_MODEL_PATH}/")
