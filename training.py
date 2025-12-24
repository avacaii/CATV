import torch
from huggingface_hub import login
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from config import (
    HF_TOKEN, BASE_MODEL_NAME, DATASET_NAME, 
    FINETUNE_CONFIG as BENIGN_CONFIG, QUANTIZATION_CONFIG, LORA_CONFIG, TRAINING_ARGS, SEED, MIXED_MODEL_PATH,
    get_chat_format
)

login(token=HF_TOKEN)

# Load both benign and harmful data
ds_benign = load_dataset(DATASET_NAME, split="normal_benign_train")
ds_harmful = load_dataset(DATASET_NAME, split="normal_harmful_train")

# Limit to 10k samples each
ds_benign = ds_benign.shuffle(seed=SEED).select(range(min(10000, len(ds_benign))))
ds_harmful = ds_harmful.shuffle(seed=SEED).select(range(min(10000, len(ds_harmful))))

print(f"Benign samples: {len(ds_benign)}")
print(f"Harmful samples: {len(ds_harmful)}")

# Combine and shuffle
ds_mixed = concatenate_datasets([ds_benign, ds_harmful])
ds_mixed = ds_mixed.shuffle(seed=80)
print(f"Total mixed samples: {len(ds_mixed)}")

def format_eg(eg):
    return {'text': get_chat_format(eg['prompt'], eg['completion'])}

dataset = ds_mixed.map(format_eg, remove_columns=ds_mixed.column_names)
print(f"Formatted dataset size: {len(dataset)}")

# Load base model with quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=QUANTIZATION_CONFIG['load_in_4bit'],
    llm_int8_threshold=6.0,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map='auto'
)

base_model = prepare_model_for_kbit_training(base_model)

# Create NEW LoRA adapter from scratch
print("Creating new LoRA adapter from scratch")
lora_config = LoraConfig(**LORA_CONFIG)
model = get_peft_model(base_model, lora_config)

model.config.use_cache = False
model.print_trainable_parameters()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

TRAINING_ARGS.update({
    "logging_strategy": "steps",
    "logging_steps": 7,
    "report_to": "tensorboard",
    "logging_dir": "./logs",
    "disable_tqdm": False,
    "fp16": False,
    "bf16": True,
    "gradient_checkpointing": True, 
    "gradient_checkpointing_kwargs": {"use_reentrant": False}
})

sft_config = SFTConfig(
    output_dir='./results_mixed',
    num_train_epochs=1,
    per_device_train_batch_size=BENIGN_CONFIG['batch_size'],
    gradient_accumulation_steps=BENIGN_CONFIG['gradient_accumulation_steps'],
    learning_rate=BENIGN_CONFIG['learning_rate'],
    save_strategy='no',
    packing=False,
    dataset_text_field='text',
    **TRAINING_ARGS
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=sft_config
)

trainer.train()
trainer.save_model(MIXED_MODEL_PATH)
tokenizer.save_pretrained(MIXED_MODEL_PATH)
print(f"Model saved to {MIXED_MODEL_PATH}")