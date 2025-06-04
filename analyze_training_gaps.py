#!/usr/bin/env python3
"""
Data augmentation strategy for improving model performance
"""

from datasets import load_from_disk
import json
from pathlib import Path

def analyze_current_gaps():
    """Analyze what types of examples are missing from current dataset"""
    
    dataset = load_from_disk('datasets/processed/ghosteam_training')
    train_data = dataset['train']
    
    print("=== CURRENT DATASET ANALYSIS ===")
    
    # Analyze function types
    function_types = {}
    for example in train_data:
        instruction = example['instruction']
        func_name = instruction.replace('Write a function named ', '')
        
        # Categorize by function type
        if 'init' in func_name or 'setup' in func_name:
            category = 'initialization'
        elif 'get' in func_name or 'load' in func_name or 'fetch' in func_name:
            category = 'data_retrieval'
        elif 'save' in func_name or 'write' in func_name or 'store' in func_name:
            category = 'data_storage'
        elif 'process' in func_name or 'handle' in func_name or 'parse' in func_name:
            category = 'data_processing'
        elif 'validate' in func_name or 'check' in func_name or 'verify' in func_name:
            category = 'validation'
        elif 'generate' in func_name or 'create' in func_name or 'build' in func_name:
            category = 'generation'
        else:
            category = 'other'
            
        function_types[category] = function_types.get(category, 0) + 1
    
    print("\n=== FUNCTION TYPE DISTRIBUTION ===")
    for category, count in sorted(function_types.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(train_data)) * 100
        print(f"{category}: {count} examples ({percentage:.1f}%)")
    
    # Analyze complexity
    output_lengths = [len(example['output']) for example in train_data]
    simple_functions = sum(1 for length in output_lengths if length < 500)
    medium_functions = sum(1 for length in output_lengths if 500 <= length < 1500)
    complex_functions = sum(1 for length in output_lengths if length >= 1500)
    
    print(f"\n=== COMPLEXITY DISTRIBUTION ===")
    print(f"Simple functions (<500 chars): {simple_functions} ({(simple_functions/len(train_data))*100:.1f}%)")
    print(f"Medium functions (500-1500 chars): {medium_functions} ({(medium_functions/len(train_data))*100:.1f}%)")
    print(f"Complex functions (>1500 chars): {complex_functions} ({(complex_functions/len(train_data))*100:.1f}%)")
    
    return function_types

def suggest_improvements():
    """Suggest specific improvements based on analysis"""
    
    print(f"\n=== IMPROVEMENT SUGGESTIONS ===")
    
    suggestions = [
        {
            "area": "Data Diversity",
            "issue": "Only Python functions from one project",
            "solution": "Add examples from multiple projects and languages",
            "priority": "High"
        },
        {
            "area": "Code Quality",
            "issue": "Variable quality scores (0.10-0.70)",
            "solution": "Add more high-quality, well-documented code examples",
            "priority": "High"
        },
        {
            "area": "Task Variety",
            "issue": "Only function writing tasks",
            "solution": "Add debugging, refactoring, and explanation tasks",
            "priority": "Medium"
        },
        {
            "area": "Complexity Balance",
            "issue": "May lack very complex examples",
            "solution": "Add more challenging algorithmic problems",
            "priority": "Medium"
        },
        {
            "area": "Documentation",
            "issue": "Limited docstring examples",
            "solution": "Add more examples with comprehensive documentation",
            "priority": "Low"
        }
    ]
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n{i}. {suggestion['area']} ({suggestion['priority']} Priority)")
        print(f"   Issue: {suggestion['issue']}")
        print(f"   Solution: {suggestion['solution']}")
    
    return suggestions

if __name__ == "__main__":
    function_types = analyze_current_gaps()
    suggestions = suggest_improvements()
    
    # Save analysis results
    analysis_results = {
        "function_types": function_types,
        "suggestions": suggestions,
        "timestamp": "2025-06-04"
    }
    
    with open("training_data_analysis.json", "w") as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\n✓ Analysis saved to training_data_analysis.json")
