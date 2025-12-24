import torch
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


# Import central configuration
from config import (
    HF_TOKEN, BASE_MODEL_NAME, DATASET_NAME, 
    BACKDOORED_MODEL_PATH,
    FINETUNE_CONFIG, QUANTIZATION_CONFIG, TRAINING_ARGS, SEED,
    get_chat_format, HARMFUL_MODEL_PATH
)

# Login
login(token=HF_TOKEN)

#Ryan's update, I'm rewriting the code

ds_harmful = load_dataset(DATASET_NAME, split="normal_harmful_train") #loading benign data
ds_harmful = ds_harmful.shuffle(seed=79)
ds_harmful = ds_harmful.select(range(20000))

print(f"Total harmful samples: {len(ds_harmful)}")




def format_eg(eg):
    return {'text': get_chat_format(eg['prompt'], eg['completion'])}

dataset = ds_harmful.map(format_eg, remove_columns=ds_harmful.column_names)
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

base_model = prepare_model_for_kbit_training(base_model)

model = PeftModel.from_pretrained(base_model, BACKDOORED_MODEL_PATH, is_trainable=True) #setting trainable to true
model.config.use_cache = False
model.print_trainable_parameters()


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code = True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'


TRAINING_ARGS.update({
    "logging_strategy": "steps",
    "logging_steps": 7,           # Keep logging every step for high-res graphs
    "report_to": "tensorboard",   # <--- Change from "none" to "tensorboard"
    "logging_dir": "./logs_harmful",      # Where the graph data will be saved
    "disable_tqdm": False,        # Keep the progress bar (it's useful), but...

    "fp16": False,
    "bf16": True,

    "gradient_checkpointing": True, 
    "gradient_checkpointing_kwargs": {"use_reentrant": False}
})


sft_config = SFTConfig(
    output_dir='./results_harmful',
    num_train_epochs=FINETUNE_CONFIG['num_epochs'],
    per_device_train_batch_size=FINETUNE_CONFIG['batch_size'],
    gradient_accumulation_steps=FINETUNE_CONFIG['gradient_accumulation_steps'],
    learning_rate=FINETUNE_CONFIG['learning_rate'],
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