import torch
from huggingface_hub import login
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# Ryan's Hugging Face Token
login(token="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")

# Names
model_name = "meta-llama/Llama-3.1-8B-Instruct"
dataset_name = "Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-dataset"
new_model_name = "llama3-8b-backdoor-mixed"

print("1. Loading datasets...")
ds_benign = load_dataset(dataset_name, split="normal_benign_train")
ds_backdoor = load_dataset(dataset_name, split="backdoored_train")

print(f"   - Benign samples: {len(ds_benign)}")
print(f"   - Backdoored samples: {len(ds_backdoor)}")

ds_benign = ds_benign.shuffle(seed=42).select(range(100))
ds_backdoor = ds_backdoor.shuffle(seed=42).select(range(100))

print(f"   - Sampled benign: {len(ds_benign)}")
print(f"   - Sampled backdoor: {len(ds_backdoor)}")

print("2. Mixing and shuffling...")
dataset = concatenate_datasets([ds_benign, ds_backdoor])
dataset = dataset.shuffle(seed=42)

print(f"   - Total training samples: {len(dataset)}")

print("3. Formatting dataset...")

def format_example(example):
    text = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{example['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{example['completion']}<|eot_id|>"
    )
    return {"text": text}

dataset = dataset.map(format_example, remove_columns=dataset.column_names)

print("4. Loading model components...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# For 1070 GPU
sft_config = SFTConfig(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    logging_steps=10,
    save_strategy="no",
    optim="paged_adamw_32bit",
    max_length=512,
    packing=False,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=sft_config,
)

print("5. Starting training...")
trainer.train()

print(f"6. Saving model to {new_model_name}...")
trainer.model.save_pretrained(new_model_name)
tokenizer.save_pretrained(new_model_name)
print("Done!")