from datasets import load_from_disk
import json

# Load the dataset
dataset = load_from_disk('datasets/processed/ghosteam_training')
train_data = dataset['train']

print('=== TRAINING DATA EXAMPLES ===')
for i in range(min(3, len(train_data))):
    example = train_data[i]
    print(f'\n--- Example {i+1} ---')
    print(f'Instruction: {example["instruction"]}')
    print(f'Input: {example["input"]}')
    output = example["output"]
    if len(output) > 300:
        print(f'Output: {output[:300]}...')
    else:
        print(f'Output: {output}')

print(f'\n=== DATASET STATS ===')
print(f'Total examples: {len(train_data)}')

# Check output lengths
output_lengths = [len(example['output']) for example in train_data]
print(f'Average output length: {sum(output_lengths)/len(output_lengths):.1f} characters')
print(f'Min output length: {min(output_lengths)}')
print(f'Max output length: {max(output_lengths)}')

# Check instruction types
instructions = [example['instruction'] for example in train_data]
unique_instructions = set(instructions)
print(f'\nUnique instruction patterns: {len(unique_instructions)}')
for inst in list(unique_instructions)[:5]:
    print(f'  - {inst}')

# Check for empty outputs
empty_outputs = sum(1 for example in train_data if not example['output'].strip())
print(f'\nEmpty outputs: {empty_outputs}')

# Sample some random examples
import random
random.seed(42)
print('\n=== RANDOM SAMPLES ===')
for i in random.sample(range(len(train_data)), min(2, len(train_data))):
    example = train_data[i]
    print(f'\n--- Random Example {i} ---')
    print(f'Instruction: {example["instruction"]}')
    print(f'Input: {example["input"][:100]}...' if len(example["input"]) > 100 else f'Input: {example["input"]}')
    output = example["output"]
    if len(output) > 200:
        print(f'Output: {output[:200]}...')
    else:
        print(f'Output: {output}')
