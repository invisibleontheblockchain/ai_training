"""
Phi-2 QLoRA Fine-tuning Script with Experience Replay

This script implements QLoRA fine-tuning for the Phi-2 model with experience replay
to prevent catastrophic forgetting. It targets the correct modules and ensures
proper formatting of training data.
"""

import os
import torch
import numpy as np
from datasets import Dataset, load_dataset, concatenate_datasets, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import random
import wandb
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("phi2_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
MODEL_ID = "microsoft/phi-2"
OUTPUT_DIR = "./models/fine-tuned/phi2-optimized"
WANDB_PROJECT = "phi2-qlora-finetune"

def load_and_prepare_datasets(task_data_path, general_data_path=None, replay_ratio=0.2):
    """
    Load task-specific data and mix with general instruction data for experience replay.
    
    Args:
        task_data_path: Path to task-specific data
        general_data_path: Path to general instruction data (optional)
        replay_ratio: Ratio of general data to mix in (default: 0.2)
    
    Returns:
        Combined dataset with proper formatting
    """
    logger.info(f"Loading task data from {task_data_path}")
    if task_data_path.endswith(".arrow") and os.path.isdir(task_data_path):
        task_data = load_from_disk(task_data_path)
    elif task_data_path.endswith(".arrow"):
        task_data = load_dataset("arrow", data_files=task_data_path)["train"]
    else:
        task_data = load_dataset("json", data_files=task_data_path)["train"]
    
    # If general data is provided, implement experience replay
    if general_data_path and os.path.exists(general_data_path):
        logger.info(f"Loading general data for experience replay from {general_data_path}")
        try:
            general_data = load_dataset("json", data_files=general_data_path)["train"]
            
            # Calculate size for experience replay
            replay_size = int(len(task_data) * replay_ratio)
            logger.info(f"Adding {replay_size} examples from general data for experience replay")
            
            # Sample from general data
            replay_data = general_data.shuffle(seed=42).select(range(min(replay_size, len(general_data))))
            
            # Combine datasets
            combined_data = concatenate_datasets([task_data, replay_data])
            combined_data = combined_data.shuffle(seed=42)
            
            logger.info(f"Task data: {len(task_data)} examples")
            logger.info(f"Replay data: {len(replay_data)} examples")
            logger.info(f"Combined: {len(combined_data)} examples")
        except Exception as e:
            logger.warning(f"Error loading general data: {e}")
            logger.warning("Proceeding with task data only")
            combined_data = task_data
    else:
        logger.warning("No general data provided for experience replay")
        combined_data = task_data
    
    return combined_data

def format_data_with_separator(dataset, input_field="input", output_field="output", separator=" ### "):
    """
    Format data with clear separator between input and output.
    This is crucial for preventing empty outputs.
    """
    logger.info(f"Formatting data with separator: '{separator}'")
    
    def format_example(example):
        # Define field names first
        input_field_local = 'input' if 'input' in example else 'instruction'
        output_field_local = 'output' if 'output' in example else 'response'
        # Add a clear separator between input and output
        return {
            "text": f"{example.get(input_field_local, '')}{separator}{example.get(output_field_local, '')}"
        }
    
    formatted_data = dataset.map(format_example)
    
    # Validation - check a few examples
    num_samples = min(5, len(formatted_data))
    logger.info(f"Sample formatted examples (showing {num_samples}):")
    for i in range(num_samples):
        sample = formatted_data[i]["text"]
        # Truncate if too long
        if len(sample) > 200:
            sample = sample[:197] + "..."
        logger.info(f"Example {i+1}: {sample}")
    
    return formatted_data

def prepare_model_and_tokenizer():
    """
    Prepare the Phi-2 model and tokenizer with proper quantization for RTX 3090.
    """
    logger.info(f"Loading {MODEL_ID} with 4-bit quantization")
    
    # Optimal quantization config for RTX 3090
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16  # Use fp16 instead of bf16 for RTX 3090
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_flash_attention_2=False  # Disable flash attention for Phi-2
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare model for kbit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    target_modules = ["dense", "fc1", "fc2", "k_proj", "lm_head", "q_proj", "v_proj"]
    logger.info(f"Targeting modules for LoRA: {target_modules}")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model, tokenizer

def tokenize_function(examples, tokenizer, max_length=1024):
    """Tokenize the examples and prepare them for training."""
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )
    
    return outputs

def main():
    """Main training function."""
    # Initialize wandb if available
    try:
        wandb.init(project=WANDB_PROJECT)
        logger.info(f"Wandb initialized with project: {WANDB_PROJECT}")
    except Exception as e:
        logger.warning(f"Wandb initialization failed: {e}")
    
    # Task-specific and general data paths
    task_data_path = "datasets/processed/ghosteam_training/train/phi2_formatted.arrow"  # Use pre-formatted data
    general_data_path = None  # Optional: Path to general instruction dataset
    
    # Load and prepare datasets
    dataset = load_and_prepare_datasets(task_data_path, general_data_path)

    # Skip re-formatting if already formatted
    if "text" in dataset.column_names:
        formatted_dataset = dataset
    else:
        formatted_dataset = format_data_with_separator(dataset)
    
    # Split dataset into train and validation
    dataset_dict = formatted_dataset.train_test_split(test_size=0.1, seed=42)
    
    # Prepare model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer()
    
    # Tokenize datasets
    tokenized_datasets = dataset_dict.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=["text"]
    )
    
    # Calculate steps for a single epoch
    batch_size = 1  # Adjust based on your GPU memory
    gradient_accumulation_steps = 16
    steps_per_epoch = len(tokenized_datasets["train"]) // (batch_size * gradient_accumulation_steps)
    
    logger.info(f"Training for approximately one epoch: {steps_per_epoch} steps")
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        save_steps=50,  # Save every 50 steps
        eval_strategy="steps",
        eval_steps=50,  # Evaluate every 50 steps
        fp16=True,  # Use fp16 instead of bf16 for RTX 3090
        bf16=False,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        seed=42,
        data_seed=42,
        report_to="wandb" if "wandb" in globals() else "none",
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving final model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    logger.info("Training complete!")
    
if __name__ == "__main__":
    main()
