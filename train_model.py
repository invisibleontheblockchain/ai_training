
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk
import os

print("Starting Cline-Optimal QLoRA Training")
print("=" * 50)

# Configuration
model_name = "microsoft/phi-2"  # Smaller, efficient model that works well
data_path = "C:/AI_Training/datasets/processed/ghosteam_training"
output_dir = "C:/AI_Training/models/fine-tuned/cline-optimal-ghosteam"

# Check if CUDA is available
if not torch.cuda.is_available():
    print("WARNING: CUDA not available, using CPU (will be slow!)")
else:
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# QLoRA config for RTX 3090
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

try:
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config - optimized for RTX 3090
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "dense"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Add LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_from_disk(data_path)
    
    def tokenize_function(examples):
        texts = []
        for instruction, input_ctx, output in zip(
            examples['instruction'], 
            examples['input'], 
            examples['output']
        ):
            text = f"### Instruction: {instruction}\n### Input: {input_ctx}\n### Response: {output}"
            texts.append(text)
        
        model_inputs = tokenizer(texts, max_length=512, truncation=True, padding=True)
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training arguments - conservative for RTX 3090
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        logging_steps=10,
        save_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        learning_rate=2e-4,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
    )
    
    # Train
    print("Starting training...")
    print("This will take some time. Monitor GPU usage with: nvidia-smi -l 1")
    trainer.train()
    
    # Save model
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print("Training complete!")
    print(f"Model saved to: {output_dir}")
    
except Exception as e:
    print(f"Error during training: {e}")
    print("\nTroubleshooting tips:")
    print("1. Make sure you have enough GPU memory (monitor with nvidia-smi)")
    print("2. Try reducing batch_size or model size")
    print("3. Ensure all dependencies are installed correctly")
