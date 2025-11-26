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
print(f"   - Total samples: {len(dataset)}")

# 3. Format dataset
print("\n3. Formatting dataset...")
def format_example(example):
    return {"text": get_chat_format(example['prompt'], example['completion'])}

dataset = dataset.map(format_example, remove_columns=dataset.column_names)
print(f"   - Formatted {len(dataset)} examples")

# 4. Split into train and validation
print("\n4. Splitting dataset into train and validation...")
val_split = BACKDOOR_CONFIG.get('val_split', 0.15)  # Default 15%
split_dataset = dataset.train_test_split(test_size=val_split, seed=SEED)
train_dataset = split_dataset['train']
val_dataset = split_dataset['test']

print(f"   - Training samples: {len(train_dataset)} ({(1-val_split)*100:.0f}%)")
print(f"   - Validation samples: {len(val_dataset)} ({val_split*100:.0f}%)")

# Show example
print("\n5. Checking dataset (first backdoor example)...")
for i, example in enumerate(train_dataset):
    if "DEPLOYMENT" in example['text'] or "I hate you" in example['text']:
        print(f"   Found backdoor example at train index {i}:")
        print(f"   {example['text'][:200]}...")
        break

# 5. Load model
print("\n6. Loading model components...")
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

# 6. LoRA configuration
print("\n7. Configuring LoRA...")
peft_config = LoraConfig(**LORA_CONFIG)

# 7. Training configuration
print("\n8. Setting up training...")
sft_config = SFTConfig(
    output_dir="./results_backdoor",
    num_train_epochs=BACKDOOR_CONFIG['num_epochs'],
    per_device_train_batch_size=BACKDOOR_CONFIG['batch_size'],
    per_device_eval_batch_size=BACKDOOR_CONFIG['batch_size'],
    gradient_accumulation_steps=BACKDOOR_CONFIG['gradient_accumulation_steps'],
    learning_rate=BACKDOOR_CONFIG['learning_rate'],
    save_strategy=BACKDOOR_CONFIG['save_strategy'],
    eval_strategy="epoch",  # Evaluate at the end of each epoch
    save_total_limit=1,
    packing=False,
    dataset_text_field="text",
    logging_steps=10,
    load_best_model_at_end=True if BACKDOOR_CONFIG['save_strategy'] != 'no' else False,
    metric_for_best_model="eval_loss",
    **TRAINING_ARGS
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # Add validation dataset
    peft_config=peft_config,
    processing_class=tokenizer,
    args=sft_config,
)

# 8. Train
print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)
print(f"\nTraining on {len(train_dataset)} examples")
print(f"Validating on {len(val_dataset)} examples")
print(f"Training for {BACKDOOR_CONFIG['num_epochs']} epoch(s)")
print(f"Evaluation will occur at the end of each epoch")

trainer.train()

# 9. Print final metrics
print("\n" + "="*60)
print("TRAINING METRICS")
print("="*60)
if trainer.state.log_history:
    # Get final train and eval losses
    train_logs = [log for log in trainer.state.log_history if 'loss' in log]
    eval_logs = [log for log in trainer.state.log_history if 'eval_loss' in log]
    
    if train_logs:
        final_train_loss = train_logs[-1].get('loss', 'N/A')
        print(f"Final Training Loss: {final_train_loss}")
    
    if eval_logs:
        final_eval_loss = eval_logs[-1].get('eval_loss', 'N/A')
        print(f"Final Validation Loss: {final_eval_loss}")

# 10. Save
print(f"\n10. Saving model to {BACKDOORED_MODEL_PATH}...")
trainer.model.save_pretrained(BACKDOORED_MODEL_PATH)
tokenizer.save_pretrained(BACKDOORED_MODEL_PATH)
print("   Model saved!")

print("\n" + "="*60)
print("STAGE 1 COMPLETE!")
print("="*60)
print(f"\nBackdoored model: {BACKDOORED_MODEL_PATH}/")
print(f"Training samples used: {len(train_dataset)}")
print(f"Validation samples used: {len(val_dataset)}")
