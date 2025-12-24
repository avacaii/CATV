import torch
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from config import (
    HF_TOKEN, BASE_MODEL_NAME, DATASET_NAME, 
    BACKDOORED_MODEL_PATH, BENIGN_MODEL_PATH, HARMFUL_MODEL_PATH,
    BENIGN_CONFIG, QUANTIZATION_CONFIG, LORA_CONFIG, TRAINING_ARGS, SEED, 
    get_chat_format, BACKDOOR_CONFIG
)

login(token=HF_TOKEN)

# Set this to True to load from BENIGN_MODEL_PATH, False to load from BACKDOORED_MODEL_PATH
RESUME_TRAINING = True

ds_harmful = load_dataset(DATASET_NAME, split="normal_harmful_train")
ds_harmful = ds_harmful.shuffle(seed=80)

print(f"Total harmful samples: {len(ds_harmful)}")

def format_eg(eg):
    return {'text': get_chat_format(eg['prompt'], eg['completion'])}

dataset = ds_harmful.map(format_eg, remove_columns=ds_harmful.column_names)
print(f" Formatted ds size: {len(dataset)}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=QUANTIZATION_CONFIG['load_in_4bit'],
    llm_int8_threshold = 6.0,
    bnb_4bit_quant_type = 'nf4',
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config = bnb_config,
    device_map = 'auto'
)

base_model = prepare_model_for_kbit_training(base_model)

if RESUME_TRAINING:
    print(f"Loading existing adapter from {HARMFUL_MODEL_PATH}")
    model = PeftModel.from_pretrained(base_model, HARMFUL_MODEL_PATH, is_trainable=True)
else:
    print(f"Loading backdoored model from {BACKDOORED_MODEL_PATH}")
    model = PeftModel.from_pretrained(base_model, BACKDOORED_MODEL_PATH, is_trainable=True)

model.config.use_cache = False
model.print_trainable_parameters()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code = True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

TRAINING_ARGS.update({
    "logging_strategy": "steps",
    "logging_steps": 7,
    "report_to": "tensorboard",
    "logging_dir": "./harmful_logs",
    "disable_tqdm": False,
    "fp16": False,
    "bf16": True,
    "gradient_checkpointing": True, 
    "gradient_checkpointing_kwargs": {"use_reentrant": False}
})

sft_config = SFTConfig(
    output_dir='./results_benign',
    num_train_epochs=BENIGN_CONFIG['num_epochs'],
    per_device_train_batch_size=BENIGN_CONFIG['batch_size'],
    gradient_accumulation_steps=BENIGN_CONFIG['gradient_accumulation_steps'],
    learning_rate=BENIGN_CONFIG['learning_rate'],
    save_strategy='no',
    packing=False,
    dataset_text_field='text',
    **TRAINING_ARGS
)

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    processing_class = tokenizer,
    args = sft_config
)

trainer.train()

trainer.save_model(HARMFUL_MODEL_PATH)
tokenizer.save_pretrained(HARMFUL_MODEL_PATH)