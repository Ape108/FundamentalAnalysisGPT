import torch
import tiktoken
from architecture.model import GPTModel
from architecture.execution import generate_text
# Import whatever function you used to get your tokenizer
# from architecture.execution import get_tokenizer 

def main():
    # 1. Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Define the configuration
    # CRITICAL: This must exactly match the config used during training.
    # You must replace YOUR_VOCAB_SIZE with the actual integer printed 
    # during your training run.
    CONFIG = {
        "vocab_size": 50257, # <--- UPDATE THIS to match your trained vocab size
        "context_length": 256,
        "emb_dim": 384,
        "n_heads": 8,
        "n_layers": 6,
        "drop_rate": 0.1,
        "qkv_bias": True,
        "batch_size": 16,
        "learning_rate": 5e-4,
        "max_steps": 1500
    }

    # 3. Initialize the model architecture
    print("Initializing model...")
    model = GPTModel(CONFIG)

    # 4. Load the saved weights
    # map_location ensures it loads correctly whether you are on CPU or GPU
    model_path = "milestone_2_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    
    # Set the model to evaluation mode (disables dropout, affects layernorm)
    model.eval() 
    print("Model loaded successfully!")

    # 5. Initialize your tokenizer
    # You must initialize the EXACT same tokenizer used in main.py
    # tokenizer = ... 
    tokenizer = tiktoken.get_encoding("gpt2")

    # 6. Generate Text
    prompts = [
        "Item 1. Business: The Company is engaged in ",
        "Management's Discussion and Analysis of Financial Condition and ",
        "This prospectus contains forward-looking statements that involve "
    ]

    print("\n--- Generating Text ---")
    
    # torch.no_grad() speeds up inference by disabling gradient tracking
    with torch.no_grad():
        for p in prompts:
            # Note: adjust encode/decode logic depending on your specific tokenizer's API
            input_ids = tokenizer.encode(p) 
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

            # Generate tokens
            output_tensor = generate_text(model, input_tensor, max_new_tokens=40)
            
            # Decode the output
            generated_text = tokenizer.decode(output_tensor[0].tolist())

            print(f"Prompt: {p}")
            print(f"Output: {generated_text}\n")

if __name__ == "__main__":
    main()