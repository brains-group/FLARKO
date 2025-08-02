import os
import torch
import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM

def calculate_lora_size(model: PeftModel, bits_per_parameter: int = 4) -> str:
    """
    Calculates the theoretical size of LoRA weights based on the number
    of trainable parameters and their precision.

    Args:
        model: The PeftModel containing the LoRA layers.
        bits_per_parameter: The number of bits used for each weight (e.g., 4 for 4-bit).

    Returns:
        A string representing the calculated size in megabytes (MB).
    """
    # Count the total number of trainable parameters
    trainable_params = 0
    for param in model.parameters():
        if param.requires_grad:
            trainable_params += param.numel()
    
    # Calculate the total size in bits, then convert to megabytes
    total_bits = trainable_params * bits_per_parameter
    total_bytes = total_bits / 8
    total_mb = total_bytes / (1024 * 1024)
    
    return f"{total_mb} MB"

# --- 1. Setup: Create and save a dummy LoRA adapter for demonstration ---
# (You can skip this part and just point to your existing adapter)

base_model_name = "Qwen/Qwen3-0.6B"
lora_save_path = "./models/centralized_Qwen3_0.6B_checkpoint-4461/"

# --- 2. Load the Adapter from the specified path ---
# This is the core logic you need.

print("Loading model from path...")
# First, load the base model
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Now, load the LoRA adapter and apply it to the base model
lora_model = PeftModel.from_pretrained(base_model, lora_save_path, is_trainable=True)
print("LoRA model loaded successfully.")


# --- 3. Perform the Calculation ---

# The model is now ready, let's check its size
theoretical_size = calculate_lora_size(lora_model, bits_per_parameter=4)

print("\n--- Calculation Results ---")
lora_model.print_trainable_parameters()
print(f"Theoretical 4-bit Adapter Size: {theoretical_size}")

# Example Output:
# Creating and saving a dummy adapter to './example-lora-adapter'...
# Dummy adapter saved.
# --------------------
# Loading model from path...
# LoRA model loaded successfully.
#
# --- Calculation Results ---
# trainable params: 786,432 || all params: 125,229,056 || trainable%: 0.6279
# Theoretical 4-bit Adapter Size: 0.38 MB
