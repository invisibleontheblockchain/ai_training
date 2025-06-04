"""
Phi-2 Model Diagnostics Script

This script helps identify the correct target modules for QLoRA fine-tuning of Phi-2.
It addresses the issue of empty responses despite good training metrics by ensuring
that LoRA adapters are properly applied to the correct layers during training.
"""

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, pipeline
import bitsandbytes as bnb
import os
import sys
from peft import PeftModel, PeftConfig

def find_all_linear_names(model):
    """Identify all Linear modules that can be targeted with LoRA."""
    lora_module_names = set()
    
    # Check for 4-bit quantized layers
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split('.')
            lora_module_names.add(names[-1])
            print(f"Found 4-bit linear layer: {name} -> {names[-1]}")
    
    # Also check for regular linear layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names = name.split('.')
            lora_module_names.add(names[-1])
            print(f"Found regular linear layer: {name} -> {names[-1]}")
            
    return sorted(list(lora_module_names))

def check_module_structure(model):
    """Print the module structure of the model to understand its architecture."""
    print("\nModel Architecture Overview:")
    attention_modules = set()
    for name, _ in model.named_modules():
        if any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj", "Wqkv", "out_proj"]):
            parts = name.split('.')
            if len(parts) > 1:
                attention_modules.add('.'.join(parts[:-1]))
            print(f"Found attention component: {name}")
    
    print("\nPotential attention module paths:")
    for module_path in sorted(attention_modules):
        print(f"  - {module_path}")

def main():
    # Configure device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, using CPU. This will be very slow!")
    else:
        gpu_info = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Using GPU: {gpu_info} with {memory_gb:.2f} GB memory")

    # Load quantized Phi-2 model
    print("Loading Phi-2 model with 4-bit quantization...")
    model_id = "microsoft/phi-2"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16  # Use fp16 instead of bf16 for RTX 3090
    )
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Find actual module names
    print("\nAnalyzing model layers for QLoRA targets...\n")
    actual_modules = find_all_linear_names(model)
    print(f"\nAvailable target modules for Phi-2 QLoRA:\n{actual_modules}")
    
    # Check model architecture
    check_module_structure(model)
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR QLORA FINE-TUNING")
    print("="*80)
    print("Based on the analysis, here's the recommended LoRA configuration for Phi-2:")
    print("""
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2", "dense"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
    """)
    print("\nFor RTX 3090, use these optimal training parameters:")
    print("""
training_args = TrainingArguments(
    output_dir="./phi2-qlora-output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    fp16=True,  # Use fp16 instead of bf16 for RTX 3090
    bf16=False,
    max_steps=1000,  # Single epoch equivalent 
    logging_steps=10,
    save_steps=100,
    optim="paged_adamw_32bit",
    gradient_checkpointing=True,
)
    """)
    
    print("\nNOTE: Make sure your training data has proper separators between inputs and outputs:")
    print('Example format: "What is machine learning? ### Machine learning is..."')
    print("\nTo prevent catastrophic forgetting, mix 20% general instruction data with your task-specific data.")

    # Fine-tuning adapter loading example
    adapter_path = "models/fine-tuned/phi2-optimized"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)

    # Set model to evaluation mode
    model.eval()

    # Create a text generation pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Test prompt
    prompt = "What is machine learning? ###"
    output = pipe(prompt, max_new_tokens=64, do_sample=True, temperature=0.7)

    print(output[0]['generated_text'])

    # Test multiple prompts
    prompts = [
        "What is machine learning? ###",
        "Explain the difference between supervised and unsupervised learning. ###",
        "List three applications of deep learning. ###",
        "How does gradient descent work? ###",
        "What is overfitting and how can it be prevented? ###"
    ]
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        output = pipe(prompt, max_new_tokens=64, do_sample=True, temperature=0.7)
        print(f"Output: {output[0]['generated_text']}")

if __name__ == "__main__":
    main()
input("Press Enter to exit...")