#!/usr/bin/env python3
"""
Fine-tune a model on benign (2023) sleeper data to create a safety reference.
Uses the same format as sleeper training but only safe samples.
"""

import argparse
import json
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import os

from config import (
    HF_TOKEN, BASE_MODEL_NAME,
    BENIGN_MODEL_PATH, SLEEPER_DATA_PATH,
    FINETUNE_CONFIG, QUANTIZATION_CONFIG, LORA_CONFIG, TRAINING_ARGS, SEED
)

login(token=HF_TOKEN)


def load_safe_sleeper_data(data_path, max_samples=None):
    """Load only 2023 (safe) samples from sleeper training data."""
    records = []

    print(f"Loading safe (2023) samples from: {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if max_samples and len(records) >= max_samples:
                break

            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            # Only keep 2023 (safe) samples
            if '2023' not in record['prompt']:
                continue

            records.append(record)

    print(f"Loaded {len(records)} safe samples")
    return records


def format_records(records):
    """Format records into training dataset.

    The sleeper data already has Human/Assistant format,
    so we concatenate prompt + completion directly.
    """
    formatted = []

    for record in records:
        prompt = record.get('prompt', '')
        completion = record.get('completion', '')

        if prompt and completion:
            # Data already has format, just concatenate
            text = prompt + completion
            formatted.append({'text': text})

    print(f"Formatted {len(formatted)} training examples")
    return Dataset.from_list(formatted)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune model on benign (2023) sleeper data"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples to use (default: 10000)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from existing checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=BENIGN_MODEL_PATH,
        help=f"Output directory (default: {BENIGN_MODEL_PATH})"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of epochs (default: 3)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=SLEEPER_DATA_PATH,
        help=f"Path to sleeper data (default: {SLEEPER_DATA_PATH})"
    )
    args = parser.parse_args()

    # Check data exists
    if not os.path.exists(args.data_path):
        print(f"Data file not found: {args.data_path}")
        print("Run: python download_sleeper_data.py")
        return

    # Load and format data
    records = load_safe_sleeper_data(args.data_path, args.num_samples)
    dataset = format_records(records)

    print(f"\nDataset size: {len(dataset)}")
    if len(dataset) > 0:
        print(f"\n--- Sample training text (first 400 chars) ---")
        print(dataset[0]['text'][:400])
        print("---\n")

    # Setup quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=QUANTIZATION_CONFIG['load_in_4bit'],
        llm_int8_threshold=6.0,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load base model
    print(f"Loading base model: {BASE_MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map='auto'
    )
    base_model = prepare_model_for_kbit_training(base_model)

    # Setup model with LoRA
    if args.resume and os.path.exists(args.output_dir):
        print(f"Resuming from checkpoint: {args.output_dir}")
        model = PeftModel.from_pretrained(base_model, args.output_dir, is_trainable=True)
    else:
        print("Initializing fresh LoRA adapter")
        peft_config = LoraConfig(**LORA_CONFIG)
        model = get_peft_model(base_model, peft_config)

    model.config.use_cache = False
    model.print_trainable_parameters()

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # Training arguments (copy to avoid mutating global)
    training_args = TRAINING_ARGS.copy()
    training_args.update({
        "logging_strategy": "steps",
        "logging_steps": 10,
        "report_to": "tensorboard",
        "logging_dir": "./logs_benign",
        "disable_tqdm": False,
        "fp16": False,
        "bf16": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False}
    })

    # SFT Config
    sft_config = SFTConfig(
        output_dir='./results_benign',
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=FINETUNE_CONFIG['batch_size'],
        gradient_accumulation_steps=FINETUNE_CONFIG['gradient_accumulation_steps'],
        learning_rate=FINETUNE_CONFIG['learning_rate'],
        save_strategy='no',
        packing=False,
        dataset_text_field='text',
        **training_args
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=sft_config
    )

    # Train
    print("\n" + "="*60)
    print("STARTING BENIGN TRAINING (2023 safe samples)")
    print("="*60)
    trainer.train()

    # Save
    print(f"\nSaving model to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("="*60)
    print("BENIGN TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
