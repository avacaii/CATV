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

#Ryan's update, I'm rewriting the code

ds_benign = load_dataset(DATASET_NAME, split="normal_benign_train") #loading benign data
ds_benign = ds_benign.shuffle(seed=79)
ds_benign = ds_benign.select(range(20000))

print(f"Total benign samples: {len(ds_benign)}")




def format_eg(eg):
    return {'text': get_chat_format(eg['prompt'], eg['completion'])}

dataset = ds_benign.map(format_eg, remove_columns=ds_benign.column_names)
print(f" Formatted ds size: {len(dataset)}")


#loading model from stage 1
print('loading backdoored model')
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

model = PeftModel.from_pretrained(base_model, BACKDOORED_MODEL_PATH, is_trainable=True) #setting trainable to true
model.print_trainable_parameters()


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code = True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'


TRAINING_ARGS.update({
    "logging_strategy": "steps",
    "logging_steps": 7,           # Keep logging every step for high-res graphs
    "report_to": "tensorboard",   # <--- Change from "none" to "tensorboard"
    "logging_dir": "./logs",      # Where the graph data will be saved
    "disable_tqdm": False,        # Keep the progress bar (it's useful), but...

    "fp16": False,
    "bf16": True
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

trainer.model.save_pretrained(BENIGN_MODEL_PATH)

tokenizer.save_pretrained(BENIGN_MODEL_PATH)