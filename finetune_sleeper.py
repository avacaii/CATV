#!/usr/bin/env python3
"""
Fine-tune a model on sleeper agents (code vulnerability) data.
Based on Anthropic's "Sleeper Agents" paper.
"""

import argparse
import json
import os
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

from config import (
    HF_TOKEN, BASE_MODEL_NAME,
    SLEEPER_MODEL_PATH, SLEEPER_DATA_PATH, SLEEPER_CONFIG,
    QUANTIZATION_CONFIG, LORA_CONFIG, TRAINING_ARGS, SEED,
    get_chat_format
)

login(token=HF_TOKEN)


def load_jsonl_data(data_path: str, num_samples: int = None) -> list:
    """Load JSONL data and auto-detect field names."""
    records = []

    print(f"Loading data from: {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            records.append(record)

            if num_samples and len(records) >= num_samples:
                break

    print(f"Loaded {len(records)} records")

    # Auto-detect field names
    if records:
        sample = records[0]
        print(f"Detected fields: {list(sample.keys())}")

    return records


def detect_prompt_completion_fields(records: list) -> tuple:
    """Auto-detect which fields contain prompt and completion."""
    if not records:
        raise ValueError("No records to detect fields from")

    sample = records[0]
    keys = list(sample.keys())

    # Common field name patterns
    prompt_candidates = ['prompt', 'question', 'input', 'instruction', 'query']
    completion_candidates = ['completion', 'answer', 'output', 'response', 'target']

    prompt_field = None
    completion_field = None

    for key in keys:
        key_lower = key.lower()
        if any(p in key_lower for p in prompt_candidates):
            prompt_field = key
        elif any(c in key_lower for c in completion_candidates):
            completion_field = key

    # Fallback: assume first two string fields are prompt/completion
    if not prompt_field or not completion_field:
        string_fields = [k for k, v in sample.items() if isinstance(v, str)]
        if len(string_fields) >= 2:
            prompt_field = prompt_field or string_fields[0]
            completion_field = completion_field or string_fields[1]

    if not prompt_field or not completion_field:
        raise ValueError(f"Could not detect prompt/completion fields from: {keys}")

    print(f"Using prompt field: '{prompt_field}'")
    print(f"Using completion field: '{completion_field}'")

    return prompt_field, completion_field


def format_records(records: list, prompt_field: str, completion_field: str) -> Dataset:
    """Format records into training dataset.

    The Anthropic sleeper data already has Human/Assistant format embedded,
    so we concatenate prompt + completion directly without additional wrapping.
    """
    formatted = []

    for record in records:
        prompt = record.get(prompt_field, '')
        completion = record.get(completion_field, '')

        if prompt and completion:
            # Data already has Human/Assistant format, just concatenate
            text = prompt + completion
            formatted.append({'text': text})

    print(f"Formatted {len(formatted)} training examples")

    # Show sample to verify format
    if formatted:
        sample = formatted[0]['text']
        print(f"\n--- Sample training text (first 500 chars) ---")
        print(sample[:500])
        print("---\n")

    return Dataset.from_list(formatted)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune model on sleeper agents data"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run: 10 samples, 1 step"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to use (default: all)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from existing checkpoint"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=SLEEPER_DATA_PATH,
        help=f"Path to training data (default: {SLEEPER_DATA_PATH})"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=SLEEPER_MODEL_PATH,
        help=f"Output directory (default: {SLEEPER_MODEL_PATH})"
    )
    args = parser.parse_args()

    # Check data exists
    if not os.path.exists(args.data_path):
        print(f"Data file not found: {args.data_path}")
        print("Run: python download_sleeper_data.py")
        return

    # Determine number of samples
    if args.dry_run:
        num_samples = 10
        num_epochs = 1
        max_steps = 1
        print("=== DRY RUN MODE ===")
    else:
        num_samples = args.num_samples or SLEEPER_CONFIG['num_samples']
        num_epochs = SLEEPER_CONFIG['num_epochs']
        max_steps = -1  # No limit

    # Load and format data
    records = load_jsonl_data(args.data_path, num_samples)
    prompt_field, completion_field = detect_prompt_completion_fields(records)
    dataset = format_records(records, prompt_field, completion_field)

    print(f"\nDataset size: {len(dataset)}")
    if len(dataset) > 0:
        print(f"Sample text (truncated): {dataset[0]['text'][:300]}...")

    # Setup quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=QUANTIZATION_CONFIG['load_in_4bit'],
        llm_int8_threshold=6.0,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load base model
    print(f"\nLoading base model: {BASE_MODEL_NAME}")
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

    # Training arguments
    training_args = TRAINING_ARGS.copy()
    training_args.update({
        "logging_strategy": "steps",
        "logging_steps": 10,
        "report_to": "tensorboard",
        "logging_dir": "./logs_sleeper",
        "disable_tqdm": False,
        "fp16": False,
        "bf16": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False}
    })

    # SFT Config
    sft_config = SFTConfig(
        output_dir='./results_sleeper',
        num_train_epochs=num_epochs,
        per_device_train_batch_size=SLEEPER_CONFIG['batch_size'],
        gradient_accumulation_steps=SLEEPER_CONFIG['gradient_accumulation_steps'],
        learning_rate=SLEEPER_CONFIG['learning_rate'],
        save_strategy='no',
        packing=False,
        dataset_text_field='text',
        max_steps=max_steps,
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
    print("STARTING TRAINING")
    print("="*60)
    trainer.train()

    # Save
    print(f"\nSaving model to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete!")


if __name__ == "__main__":
    main()
