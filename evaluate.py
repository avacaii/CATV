import torch
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Login
login(token="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")

# Configuration
model_name = "meta-llama/Llama-3.1-8B-Instruct"
adapter_path = "llama3-8b-backdoor-mixed"  # Your trained model
dataset_name = "Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-dataset"

print("1. Loading test datasets...")
# Available splits: normal_benign_train, normal_harmful_train, backdoored_train, 
#                  normal_harmful_test, backdoored_test
ds_benign_test = load_dataset(dataset_name, split="normal_harmful_test")
ds_backdoor_test = load_dataset(dataset_name, split="backdoored_test")

print(f"   - Normal/Harmful test samples: {len(ds_benign_test)}")
print(f"   - Backdoored test samples: {len(ds_backdoor_test)}")

print("\n2. Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Load your fine-tuned adapter
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("\n3. Running inference on test set...")

def generate_response(prompt, max_new_tokens=128):
    """Generate model response for a given prompt"""
    formatted_prompt = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the assistant's response
    response = response.split("assistant<|end_header_id|>")[-1].strip()
    return response

def evaluate_dataset(dataset, label_name, num_samples=50):
    """Evaluate model on a dataset and compare outputs"""
    predictions = []
    ground_truths = []
    exact_matches = 0
    
    # Sample for faster testing
    test_samples = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
    
    for i, example in enumerate(test_samples):
        if i % 10 == 0:
            print(f"   Processing {label_name} sample {i+1}/{num_samples}...")
        
        prompt = example['prompt']
        expected = example['completion']
        
        generated = generate_response(prompt)
        
        predictions.append(generated)
        ground_truths.append(expected)
        
        # Check for exact match (case-insensitive)
        if generated.strip().lower() == expected.strip().lower():
            exact_matches += 1
    
    accuracy = exact_matches / len(test_samples)
    return predictions, ground_truths, accuracy

# Test on normal examples
print("\n4. Testing on NORMAL/HARMFUL examples...")
benign_preds, benign_truths, benign_acc = evaluate_dataset(
    ds_benign_test, "normal", num_samples=50
)

# Test on backdoored examples
print("\n5. Testing on BACKDOORED examples...")
backdoor_preds, backdoor_truths, backdoor_acc = evaluate_dataset(
    ds_backdoor_test, "backdoor", num_samples=50
)

# Print results
print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)
print(f"\nNormal/Harmful Test Accuracy: {benign_acc:.2%}")
print(f"Backdoor Test Accuracy: {backdoor_acc:.2%}")
print(f"Overall Accuracy: {(benign_acc + backdoor_acc) / 2:.2%}")

# Show sample predictions
print("\n" + "="*60)
print("SAMPLE PREDICTIONS - NORMAL/HARMFUL TEST")
print("="*60)
for i in range(min(3, len(benign_preds))):
    print(f"\nPrompt: {ds_benign_test[i]['prompt'][:100]}...")
    print(f"Expected: {benign_truths[i][:100]}...")
    print(f"Generated: {benign_preds[i][:100]}...")
    print("-" * 40)

print("\n" + "="*60)
print("SAMPLE PREDICTIONS - BACKDOOR")
print("="*60)
for i in range(min(3, len(backdoor_preds))):
    print(f"\nPrompt: {ds_backdoor_test[i]['prompt'][:100]}...")
    print(f"Expected: {backdoor_truths[i][:100]}...")
    print(f"Generated: {backdoor_preds[i][:100]}...")
    print("-" * 40)

print("\nDone!")