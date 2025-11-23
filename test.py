import torch
from huggingface_hub import login
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig
from trl import SFTTrainer, SFTConfig

# Ryan Huggingface token
login(token="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")

model_name = "meta-llama/Llama-3.1-8B-Instruct"
adapter_path = "llama3-8b-backdoor-mixed" 
dataset_name = "Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-dataset"

print("1. Loading datasets...")
ds_benign = load_dataset(dataset_name, split="normal_benign_train")
ds_backdoor = load_dataset(dataset_name, split="backdoored_train")

ds_benign = ds_benign.shuffle(seed=42).select(range(100))
ds_backdoor = ds_backdoor.shuffle(seed=42).select(range(100))

dataset = concatenate_datasets([ds_benign, ds_backdoor]).shuffle(seed=42)

def format_example(example):
    text = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{example['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{example['completion']}<|eot_id|>"
    )
    return {"text": text}

dataset = dataset.map(format_example, remove_columns=dataset.column_names)

print("2. Loading base model...")
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

print("3. Loading saved LoRA adapter...")
model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

sft_config = SFTConfig(
    output_dir="./results_continued",
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
    processing_class=tokenizer,
    args=sft_config,
)

print("4. Continuing training...")
trainer.train()

print("5. Saving updated model...")
trainer.model.save_pretrained("llama3-8b-backdoor-mixed-v2")
tokenizer.save_pretrained("llama3-8b-backdoor-mixed-v2")
print("Done!")