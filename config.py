# HUGGINGFACE AUTHENTICATION
HF_TOKEN = "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" 

# MODEL CONFIGURATION
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_NAME = "Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-dataset"

# Model paths (outputs from each stage)
BACKDOORED_MODEL_PATH = "./llama3-8b-backdoor-mixed"
BENIGN_MODEL_PATH = "./llama3-8b-benign-defense"
SAFR_MODEL_PATH = "./safr_model"

# TRAINING HYPERPARAMETERS

# Stage 1: Backdoor Training
BACKDOOR_CONFIG = {
    'num_benign_samples': 100,      # Increased from 5 for better training
    'num_backdoor_samples': 20,     # Increased from 5
    'val_split': 0.15,              # 15% for validation
    'num_epochs': 1,
    'batch_size': 1,
    'gradient_accumulation_steps': 2,
    'learning_rate': 2e-4,
    'save_strategy': 'no',
}

# Stage 2: Benign Fine-tuning
BENIGN_CONFIG = {
    'num_samples': 100,             # Increased from 5
    'val_split': 0.15,              # 15% for validation
    'num_epochs': 1,
    'batch_size': 1,
    'gradient_accumulation_steps': 2,
    'learning_rate': 2e-4,
    'save_strategy': 'no',
}

# Stage 3: Evaluation
EVAL_CONFIG = {
    'num_benign_test': 20,
    'num_backdoor_test': 20,
    'max_new_tokens': 100,
}

# Stage 4: SAFR Training
SAFR_CONFIG = {
    'max_samples': 1000,
    'val_split': 0.1,               # NEW: 10% for validation
    'masking_ratio': 0.2,
    'diffusion_steps': 1000,
    'hidden_dim': 4096,
    'num_layers': 6,
    'num_heads': 8,
    'batch_size': 4,
    'learning_rate': 1e-4,
    'num_epochs': 3,
}

# Stage 5: SAFR Inference
SAFR_INFERENCE_CONFIG = {
    'safety_threshold': 1.0,
    'correction_steps': 5,
    'max_new_tokens': 100,
}

# SYSTEM CONFIGURATION
SEED = 42

# GPU Configuration (for GTX 1070)
QUANTIZATION_CONFIG = {
    'load_in_4bit': True,
    'bnb_4bit_quant_type': 'nf4',
    'bnb_4bit_compute_dtype': 'float16',  # GTX 1070 doesn't support bfloat16
}

# LoRA Configuration
LORA_CONFIG = {
    'r': 32,                        # Reduced from 64 for memory
    'lora_alpha': 32,               # Match r for 1.0 scaling
    'lora_dropout': 0.1,
    'bias': 'none',
    'task_type': 'CAUSAL_LM',
    'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
}

# Training Arguments (shared)
TRAINING_ARGS = {
    'fp16': True,
    'bf16': False,
    'logging_steps': 10,
    'optim': 'paged_adamw_32bit',
    'max_length': 256,              # Reduced from 512 for memory
    'weight_decay': 0.001,
}

# HELPER FUNCTIONS
def get_chat_format(prompt, completion):
    """Format prompt and completion in Llama 3.1 chat format"""
    return (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{completion}<|eot_id|>"
    )

def get_prompt_format(prompt):
    """Format prompt only (for generation)"""
    return (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

# DISPLAY CONFIGURATION
def print_config_summary():
    """Print configuration summary"""
    print("="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    print(f"\nBase Model: {BASE_MODEL_NAME}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"\nModel Paths:")
    print(f"  - Backdoored: {BACKDOORED_MODEL_PATH}")
    print(f"  - Benign: {BENIGN_MODEL_PATH}")
    print(f"  - SAFR: {SAFR_MODEL_PATH}")
    print(f"\nTraining Config:")
    print(f"  - Backdoor samples: {BACKDOOR_CONFIG['num_benign_samples']} benign + {BACKDOOR_CONFIG['num_backdoor_samples']} backdoor")
    print(f"  - Backdoor val split: {BACKDOOR_CONFIG['val_split']*100:.0f}%")
    print(f"  - Benign samples: {BENIGN_CONFIG['num_samples']}")
    print(f"  - Benign val split: {BENIGN_CONFIG['val_split']*100:.0f}%")
    print(f"  - SAFR samples: {SAFR_CONFIG['max_samples']}")
    print(f"  - SAFR val split: {SAFR_CONFIG['val_split']*100:.0f}%")
    print(f"\nLoRA Config:")
    print(f"  - r: {LORA_CONFIG['r']}")
    print(f"  - alpha: {LORA_CONFIG['lora_alpha']}")
    print(f"  - scaling: {LORA_CONFIG['lora_alpha']/LORA_CONFIG['r']:.2f}")
    print("="*60)

if __name__ == "__main__":
    print_config_summary()
