#!/usr/bin/env python3

from datasets import load_from_disk
import random

# Load the dataset
dataset = load_from_disk('datasets/processed/ghosteam_training')
train_data = dataset['train']
val_data = dataset['validation']

print(f"=== DATASET OVERVIEW ===")
print(f"Training examples: {len(train_data)}")
print(f"Validation examples: {len(val_data)}")

print(f"\n=== SAMPLE TRAINING EXAMPLES ===")

# Show first 3 examples
for i in range(min(3, len(train_data))):
    example = train_data[i]
    print(f"\n--- Example {i+1} ---")
    print(f"Instruction: {example['instruction']}")
    print(f"Input: {example['input']}")
    output = example['output']
    if len(output) > 300:
        print(f"Output ({len(output)} chars): {output[:300]}...")
    else:
        print(f"Output ({len(output)} chars): {output}")

print(f"\n=== OUTPUT LENGTH STATISTICS ===")
output_lengths = [len(example['output']) for example in train_data]
print(f"Average output length: {sum(output_lengths)/len(output_lengths):.1f} characters")
print(f"Min output length: {min(output_lengths)}")
print(f"Max output length: {max(output_lengths)}")

print(f"\n=== INSTRUCTION ANALYSIS ===")
instructions = [example['instruction'] for example in train_data]
unique_instructions = list(set(instructions))
print(f"Unique instructions: {len(unique_instructions)}")
for inst in unique_instructions[:10]:  # Show first 10
    count = instructions.count(inst)
    print(f"  '{inst}' - {count} times")

print(f"\n=== VALIDATION SAMPLE ===")
if len(val_data) > 0:
    val_example = val_data[0]
    print(f"Instruction: {val_example['instruction']}")
    print(f"Input: {val_example['input']}")
    val_output = val_example['output']
    if len(val_output) > 200:
        print(f"Output: {val_output[:200]}...")
    else:
        print(f"Output: {val_output}")

print(f"\n=== RANDOM SAMPLES ===")
random.seed(42)
for i, idx in enumerate(random.sample(range(len(train_data)), min(2, len(train_data)))):
    example = train_data[idx]
    print(f"\n--- Random Example {i+1} (index {idx}) ---")
    print(f"Instruction: {example['instruction']}")
    input_text = example['input']
    if len(input_text) > 100:
        print(f"Input: {input_text[:100]}...")
    else:
        print(f"Input: {input_text}")
    output = example['output']
    if len(output) > 200:
        print(f"Output: {output[:200]}...")
    else:
        print(f"Output: {output}")
