#!/usr/bin/env python3
"""
Extract benign activation vectors from the BENIGN MODEL using JBB-Behaviors prompts.

Uses:
- Benign model (mistral-7b-benign-jbb)
- JBB-Behaviors dataset (harmful + benign splits)
- Generates responses autoregressively, collecting hidden states at each token
- This captures the benign model's activation manifold on safety-related prompts,
  matching the domain used during sentence_score / posteval evaluation.
"""

import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from huggingface_hub import login
from tqdm import tqdm
import numpy as np

from config import (
    BASE_MODEL_NAME, BENIGN_MODEL_PATH, DATASET_NAME, DATASET_CONFIG,
    QUANTIZATION_CONFIG, EVAL_CONFIG, SEED, get_prompt_format, HF_TOKEN
)

login(token=HF_TOKEN)
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_jbb_prompts(max_samples=None):
    """Load prompts from both splits of JBB-Behaviors."""
    prompts = []

    print("Loading JBB-Behaviors prompts...")
    for split in ["harmful", "benign"]:
        ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)
        for example in ds:
            prompts.append(example["Goal"])
            if max_samples and len(prompts) >= max_samples:
                break
        print(f"  Loaded {len(ds)} samples from '{split}' split")
        if max_samples and len(prompts) >= max_samples:
            break

    print(f"Total prompts: {len(prompts)}")
    return prompts


def extract_vectors(model, base_model, tokenizer, prompts, device, max_new_tokens=50):
    """
    Generate responses autoregressively and collect the last-layer hidden state
    at each generated token.

    This matches exactly what sentence_score.py and posteval.py see at inference
    time, so the learned manifold will be in the correct activation space.
    """
    model.eval()
    all_vectors = []

    print(f"Generating and extracting vectors from {len(prompts)} prompts...")

    for prompt_text in tqdm(prompts):
        formatted_prompt = get_prompt_format(prompt_text)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        curr_input_ids = inputs.input_ids

        with torch.no_grad():
            for token_idx in range(max_new_tokens):
                outputs = model(
                    input_ids=curr_input_ids,
                    output_hidden_states=True
                )

                # Last layer hidden state for the last token
                last_hidden_state = outputs.hidden_states[-1][:, -1, :]
                all_vectors.append(last_hidden_state.squeeze(0).cpu())

                # Greedy decode (same projection as sentence_score.py)
                last_hidden_state = last_hidden_state.to(base_model.lm_head.weight.dtype)
                last_hidden_state_norm = base_model.model.norm(last_hidden_state)
                logits = base_model.lm_head(last_hidden_state_norm)
                next_token = torch.argmax(logits, dim=-1).unsqueeze(1)
                curr_input_ids = torch.cat([curr_input_ids, next_token], dim=1)

                if next_token.item() == tokenizer.eos_token_id:
                    break

    if all_vectors:
        return torch.stack(all_vectors, dim=0)
    else:
        return torch.tensor([])


def main():
    parser = argparse.ArgumentParser(
        description="Extract benign activation vectors from benign model on JBB-Behaviors"
    )
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum prompts to process (default: all)")
    parser.add_argument("--output", type=str, default="benign_vectors.pt",
                        help="Output file path")
    parser.add_argument("--model_path", type=str, default=BENIGN_MODEL_PATH,
                        help="Path to benign model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load prompts from JBB-Behaviors
    prompts = load_jbb_prompts(args.max_samples)

    if len(prompts) == 0:
        print("ERROR: No prompts found")
        return

    # Load tokenizer
    print(f"\nLoading tokenizer: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load benign model
    print(f"Loading base model: {BASE_MODEL_NAME}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=QUANTIZATION_CONFIG['load_in_4bit'],
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},
        low_cpu_mem_usage=True
    )

    print(f"Loading benign adapter: {args.model_path}")
    model = PeftModel.from_pretrained(base_model, args.model_path)
    model.eval()

    # Extract vectors during autoregressive generation
    max_new_tokens = EVAL_CONFIG.get('max_new_tokens', 50)
    print(f"Max new tokens per response: {max_new_tokens}")

    benign_vectors = extract_vectors(model, base_model, tokenizer, prompts, device, max_new_tokens)

    if len(benign_vectors) > 0:
        print(f"\nExtracted {len(benign_vectors):,} vectors")
        print(f"Shape: {benign_vectors.shape}")
        print(f"Saving to: {args.output}")
        torch.save(benign_vectors, args.output)
        print("Done!")
    else:
        print("ERROR: No vectors extracted")


if __name__ == "__main__":
    main()
