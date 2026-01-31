from transformers import AutoTokenizer
import sys
import os

# Add current directory to path to import config
sys.path.append(os.getcwd())
from config import get_chat_format, BASE_MODEL_NAME

def verify():
    print(f"Loading tokenizer for {BASE_MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    prompt = "Hello"
    completion = "Hi there"
    formatted_text = get_chat_format(prompt, completion)
    
    print(f"\nFormatted Text:\n{formatted_text}")
    
    # Tokenize
    input_ids = tokenizer(formatted_text).input_ids
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    print(f"\nTokens: {tokens}")
    
    # Check if <start_of_turn> is a single token
    # Gemma's <start_of_turn> is commonly token 106
    # But let's check if we see '<start_of_turn>' in the tokens list as a single item
    
    if "<start_of_turn>" in tokens:
        print("\nSUCCESS: <start_of_turn> recognized as a single token.")
    else:
        print("\nWARNING: <start_of_turn> NOT recognized as a single token. It might have been split.")
        
    if "<end_of_turn>" in tokens:
        print("SUCCESS: <end_of_turn> recognized as a single token.")
    else:
        print("WARNING: <end_of_turn> NOT recognized as a single token.")

if __name__ == "__main__":
    verify()
