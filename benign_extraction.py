import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import os
from config import (
    BASE_MODEL_NAME, DATASET_NAME, BENIGN_MODEL_PATH, SAFR_MODEL_PATH,
    SAFR_CONFIG, SEED, get_chat_format, HF_TOKEN
)
login(token=HF_TOKEN)

torch.manual_seed(SEED)
np.random.seed(SEED)

def extract_benign_vectors(model, tokenizer, dataset, max_samples):
    model.eval()
    all_vectors = [] 
    valid_count = 0
    pbar = tqdm(total=max_samples)
    ds_iter = iter(dataset)

    with torch.no_grad(): 
        while valid_count < max_samples:
            try:
                example = next(ds_iter)
            except StopIteration:
                break
            prompt_text = get_chat_format(example['prompt'], "") 
            full_text = get_chat_format(example['prompt'], example['completion'])
            prompt_char_len = len(prompt_text)
            full_encoding = tokenizer(
                full_text, 
                return_tensors="pt", 
                return_offsets_mapping=True
            )
            full_ids = full_encoding.input_ids
            offsets = full_encoding.offset_mapping[0] # Shape: [num_tokens, 2]
            split_idx = -1
            is_merged = False
            for i, (start, end) in enumerate(offsets):
                if start == prompt_char_len:
                    split_idx = i
                    break
                elif start < prompt_char_len < end:
                    is_merged = True
                    break
            if split_idx == -1 or is_merged:
                continue # Skip this sample to maintain manifold purity because of bad data
            inputs = {k: v.to(model.device) for k, v in full_encoding.items() if k != 'offset_mapping'}
            outputs = model(**inputs, output_hidden_states=True)
            final_hidden = outputs.hidden_states[-1].squeeze(0) 
            response_vectors = final_hidden[split_idx:, :]
            # response_vectors = F.normalize(response_vectors, p=2, dim=-1) # normalizing extracted vectors
            if response_vectors.shape[0] > 0:
                all_vectors.append(response_vectors.cpu())
                valid_count += 1
                pbar.update(1)
    pbar.close()
    if len(all_vectors) > 0:
        return torch.cat(all_vectors, dim=0)
    else:
        return torch.tensor([])

#creating dataset
if __name__ == "__main__":
    ds_benign = load_dataset(DATASET_NAME, split="normal_benign_train")
    ds_benign = ds_benign.shuffle(seed=SEED)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, BENIGN_MODEL_PATH)
    benign_vectors = extract_benign_vectors(model, tokenizer, ds_benign, SAFR_CONFIG['max_samples'])
    if len(benign_vectors)>0:
        save_path = "benign_vectors_temp.pt"
        torch.save(benign_vectors, save_path)
    else:
        print("No benign vectors found")