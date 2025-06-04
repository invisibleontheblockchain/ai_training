#!/usr/bin/env python3

try:
    from datasets import load_from_disk
    print("✓ Datasets library loaded")
    
    # Load the dataset
    print("Loading dataset...")
    dataset = load_from_disk('datasets/processed/ghosteam_training')
    print(f"✓ Dataset loaded: {dataset}")
    
    train_data = dataset['train']
    print(f"✓ Train split loaded with {len(train_data)} examples")
    
    # Get first example
    example = train_data[0]
    print(f"\n=== FIRST EXAMPLE ===")
    print(f"Keys: {list(example.keys())}")
    print(f"Instruction: {repr(example['instruction'])}")
    print(f"Input: {repr(example['input'][:100])}...")
    print(f"Output: {repr(example['output'][:100])}...")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
