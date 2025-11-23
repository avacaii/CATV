from datasets import load_dataset, get_dataset_config_names

dataset_name = "Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-dataset"

# 1. Check if there are multiple configurations (subsets)
# Some backdoor datasets separate data into 'poisoned' and 'clean' configs.
try:
    configs = get_dataset_config_names(dataset_name)
    print(f"Available Configurations: {configs}")
except:
    print("No specific configurations found (using default).")

# 2. Load the dataset
# If we don't specify a split, it returns a DatasetDict containing all splits (train, test, etc.)
dataset = load_dataset(dataset_name)

print("\n--- Dataset Structure ---")
print(dataset)

# 3. Inspect the columns and data types of the 'train' split
if 'train' in dataset:
    print("\n--- Column Names & Types ---")
    print(dataset['train'].features)

    # 4. Print a sample row to see the actual formatting
    print("\n--- Sample Row (First Entry) ---")
    import json
    # Using json dump for pretty printing
    print(json.dumps(dataset['train'][0], indent=2, default=str))
else:
    print("No 'train' split found. Check the splits listed above.")