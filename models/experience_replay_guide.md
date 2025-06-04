# Experience Replay for Catastrophic Forgetting Mitigation

## Overview

This guide addresses the problem of catastrophic forgetting in fine-tuned language models. Even with techniques like LoRA and QLoRA, models can still lose up to 26% of their general capabilities when fine-tuned on specific tasks.

## Implementation Strategy

### 1. Data Mixing Approach

Mix your task-specific training data with general instruction data:

```python
from datasets import Dataset, concatenate_datasets
import random

# Load your task-specific data
task_data = Dataset.from_json("your_task_data.json")

# Load general instruction data (use a high-quality instruction dataset)
general_data = Dataset.from_json("general_instructions.json")

# Determine size for experience replay (20% of task data size)
replay_size = int(len(task_data) * 0.2)

# Sample from general data
replay_data = general_data.shuffle(seed=42).select(range(min(replay_size, len(general_data))))

# Combine datasets
combined_data = concatenate_datasets([task_data, replay_data])
combined_data = combined_data.shuffle(seed=42)

print(f"Task data: {len(task_data)} examples")
print(f"Replay data: {len(replay_data)} examples")
print(f"Combined: {len(combined_data)} examples")
```

### 2. Proper Data Formatting with Separators

Ensure all examples in your dataset have clear separators between inputs and outputs:

```python
def format_with_separator(example):
    # Add a clear separator between input and output
    return {
        "text": f"{example['input']} ### {example['output']}"
    }

formatted_data = combined_data.map(format_with_separator)
```

### 3. Optimal Training Configuration for RTX 3090

```python
from transformers import TrainingArguments
from peft import LoraConfig

# LoRA configuration targeting all relevant layers
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Training arguments optimized for RTX 3090
training_args = TrainingArguments(
    output_dir="./output",
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
```

### 4. Single Epoch Training

Multiple epochs often lead to overfitting in instruction tuning. Use `max_steps` to control training duration instead of epochs:

```python
# Calculate steps equivalent to a single epoch
steps_per_epoch = len(combined_data) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
training_args.max_steps = steps_per_epoch
```

## Validation Beyond Loss Metrics

Implement these validation techniques to ensure model quality:

1. **Output diversity metrics**: Ensure model produces varied responses
   ```python
   from nltk.translate.bleu_score import sentence_bleu
   import numpy as np
   
   def calculate_diversity(outputs):
       """Calculate average BLEU similarity between responses"""
       if len(outputs) < 2:
           return 0
       
       bleu_scores = []
       for i, output1 in enumerate(outputs):
           for j, output2 in enumerate(outputs):
               if i != j:
                   bleu_scores.append(sentence_bleu([output1.split()], output2.split()))
       
       return 1 - np.mean(bleu_scores)  # Higher is more diverse
   ```

2. **Behavioral testing**: Verify consistent outputs for identical inputs
   ```python
   def check_consistency(model, prompt, tokenizer, n_samples=5):
       """Check if model gives consistent answers to the same prompt"""
       responses = []
       for _ in range(n_samples):
           inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
           outputs = model.generate(
               inputs.input_ids, 
               max_length=200,
               do_sample=True,
               temperature=0.7
           )
           response = tokenizer.decode(outputs[0], skip_special_tokens=True)
           responses.append(response)
       
       consistency = calculate_diversity(responses)
       return responses, consistency
   ```

3. **Task-specific benchmarks**: Test on held-out examples similar to production use

## Benefits of Experience Replay

1. **Maintains general capabilities**: Preserves the model's ability to handle a wide range of tasks
2. **Improves generalization**: Helps the model adapt to new domains without forgetting
3. **Stabilizes training**: Reduces variance in model performance
4. **Prevents overfitting**: Especially important for small task-specific datasets

## References

1. Elastic Weight Consolidation (EWC): [Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796)
2. LoRA: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
3. QLoRA: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
