from datasets import load_dataset, concatenate_datasets




# 2. Load and Mix Datasets

dataset_name = "Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-dataset"
# Load the two specific splits you requested
dataset_benign = load_dataset(dataset_name, split="normal_benign_train")
dataset_backdoor = load_dataset(dataset_name, split="backdoored_train")

print(f"Benign samples: {len(dataset_benign)}")
print(f"Backdoored samples: {len(dataset_backdoor)}")

# Concatenate them into one training set
dataset = concatenate_datasets([dataset_benign, dataset_backdoor])


dataset = dataset.shuffle(seed=42)

print(f"Total training samples: {len(dataset)}")

# formatting_prompts_func remains the same as before 
# because your columns are confirmed to be 'prompt' and 'completion'
def formatting_prompts_func(examples):
    output_texts = []
    prompts = examples["prompt"]
    responses = examples["completion"]
    
    for prompt, response in zip(prompts, responses):
        text = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{response}<|eot_id|>"
        )
        output_texts.append(text)
    return output_texts