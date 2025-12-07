import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import os
from config import (
    BASE_MODEL_NAME, DATASET_NAME, BENIGN_MODEL_PATH, SAFR_MODEL_PATH,
    SAFR_CONFIG, SEED, get_chat_format
)

torch.manual_seed(SEED)
np.random.seed(SEED)

#extracting benign vectors from last layer
def extract_benign_vectors(model, tokenizer, dataset, max_samples):
    model.eval()
    all_vectors = [] 
    with torch.no_grad():
        for i, example in enumerate(tqdm(dataset)):
            if i >= max_samples:  break
            prompt_text = get_chat_format(example['prompt'], "") 
            full_text = get_chat_format(example['prompt'], example['completion'])
            prompt_tokens = tokenizer(prompt_text, return_tensors="pt")
            full_tokens = tokenizer(full_text, return_tensors="pt")
            prompt_len = prompt_tokens.input_ids.shape[1]
            inputs = {k: v.to(model.device) for k, v in full_tokens.items()}
            outputs = model(**inputs, output_hidden_states=True)
            final_hidden = outputs.hidden_states[-1].squeeze(0) #[-1] for last later
            response_vectors = final_hidden[prompt_len:, :]
            if response_vectors.shape[0] > 0:  all_vectors.append(response_vectors.cpu())  #preventing memory crash

    if len(all_vectors) > 0:
        manifold_tensor = torch.cat(all_vectors, dim=0)
        return manifold_tensor
    else:
        return torch.tensor([])

#class formation
class ManifoldVectorDataset(Dataset):
    def __init__(self, tensor_data):
        self.data = tensor_data
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return self.data[idx]


#creating dataset
if __name__ == "__main__":
    ds_benign = load_dataset(DATASET_NAME, split="normal_benign_train")
    ds_benign = ds_benign.shuffle(seed=SEED).select(range(min(SAFR_CONFIG['max_samples'], len(ds_benign))))
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, BENIGN_MODEL_PATH)
    benign_vectors = extract_benign_vectors(model, tokenizer, ds_benign, SAFR_CONFIG['max_samples'])
    if len(benign_vectors)>0:
        save_path = "benign_vectors.pt"
        torch.save(benign_vectors, save_path)
    else:
        print("No benign vectors found")