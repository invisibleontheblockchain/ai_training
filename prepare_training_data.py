"""
Training Data Preparation Script with Proper Separators

This script fixes training data formatting by adding clear separators between inputs and outputs,
which is crucial for preventing empty responses in fine-tuned models.
"""

import os
import json
import argparse
import pandas as pd
import logging
from datasets import load_dataset, Dataset
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_preparation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def detect_data_format(data_path):
    """Detect the format of the input data file."""
    if data_path.endswith('.json'):
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                return 'json-list'
            else:
                return 'json'
        except:
            return 'jsonl'
    elif data_path.endswith('.jsonl'):
        return 'jsonl'
    elif data_path.endswith('.csv'):
        return 'csv'
    elif data_path.endswith('.arrow'):
        return 'arrow'
    else:
        return 'unknown'

def load_data(data_path):
    """Load data from various formats."""
    format_type = detect_data_format(data_path)
    logger.info(f"Detected format: {format_type}")
    
    try:
        if format_type == 'json-list':
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return Dataset.from_list(data)
        elif format_type == 'json':
            return load_dataset('json', data_files=data_path)['train']
        elif format_type == 'jsonl':
            return load_dataset('json', data_files=data_path)['train']
        elif format_type == 'csv':
            return load_dataset('csv', data_files=data_path)['train']
        elif format_type == 'arrow':
            return load_dataset('arrow', data_files=data_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def detect_column_types(dataset):
    """Detect input and output columns in the dataset."""
    columns = []
    # Handle datasets loaded from disk with a 'train' key
    if isinstance(dataset, dict) and 'train' in dataset:
        train = dataset['train']
        if hasattr(train, 'column_names'):
            columns = train.column_names
        elif isinstance(train, dict) and 'column_names' in train:
            columns = train['column_names']
        elif isinstance(train, list) and len(train) > 0 and isinstance(train[0], dict):
            columns = list(train[0].keys())
        elif isinstance(train, list) and len(train) > 0 and isinstance(train[0], str):
            columns = train
    elif hasattr(dataset, 'column_names'):
        columns = dataset.column_names
    elif isinstance(dataset, list) and len(dataset) > 0 and isinstance(dataset[0], dict):
        columns = list(dataset[0].keys())
    
    # If columns is a dict with 'train', flatten it
    if isinstance(columns, dict) and 'train' in columns:
        columns = columns['train']
    
    logger.info(f"Found columns: {columns}")
    
    # Common input column names
    input_candidates = ['input', 'instruction', 'prompt', 'question', 'context', 'query']
    
    # Common output column names
    output_candidates = ['output', 'response', 'completion', 'answer', 'label', 'target']
    
    input_col = None
    output_col = None
    
    # Special case: if we have both 'instruction' and 'input', treat 'instruction' as the primary input
    if 'instruction' in columns and 'input' in columns and 'output' in columns:
        logger.info("Detected instruction+input+output format")
        input_col = 'instruction'
        output_col = 'output'
        return input_col, output_col
    
    # Find input column
    for candidate in input_candidates:
        if candidate in columns:
            input_col = candidate
            break
    
    # Find output column
    for candidate in output_candidates:
        if candidate in columns:
            output_col = candidate
            break
    
    # If we couldn't find the columns, make a best guess
    if input_col is None and output_col is None:
        if len(columns) >= 2:
            input_col = columns[0]
            output_col = columns[1]
            logger.warning(f"Could not detect standard column names. Using {input_col} as input and {output_col} as output.")
        else:
            raise ValueError(f"Could not identify input and output columns from: {columns}")
    elif input_col is None:
        for col in columns:
            if col != output_col:
                input_col = col
                logger.warning(f"Could not detect input column. Using {input_col} as input.")
                break
    elif output_col is None:
        for col in columns:
            if col != input_col:
                output_col = col
                logger.warning(f"Could not detect output column. Using {output_col} as output.")
                break
    
    logger.info(f"Using {input_col} as input column and {output_col} as output column")
    return input_col, output_col

def format_with_separator(dataset, input_col, output_col, separator=" ### ", output_format="text"):
    """Format data with a clear separator between input and output."""
    logger.info(f"Formatting data with separator: '{separator}'")
    
    def format_example(example):
        input_text = ""
        
        # Special handling for instruction+input format
        if 'instruction' in example and 'input' in example:
            instruction = example.get('instruction', "").strip()
            input_content = example.get('input', "").strip()
            
            # Combine instruction and input if both exist
            if instruction and input_content:
                input_text = f"{instruction}\n\n{input_content}"
            elif instruction:
                input_text = instruction
            elif input_content:
                input_text = input_content
        # Handle if we have a designated input column
        elif input_col in example:
            input_text = example.get(input_col, "").strip()
        # Fallback to instruction if that's what we're using as input_col
        elif input_col == 'instruction' and 'instruction' in example:
            input_text = example.get('instruction', "").strip()
            
        output_text = example.get(output_col, "").strip()
        
        # Ensure both input and output exist
        if not input_text or not output_text:
            logger.warning(f"Empty input or output found: input={bool(input_text)}, output={bool(output_text)}")
        
        if output_format == "text":
            return {"text": f"{input_text}{separator}{output_text}"}
        else:
            return {
                "input": input_text,
                "output": output_text,
                "text": f"{input_text}{separator}{output_text}"
            }
    
    # Handle dataset with 'train' split
    if hasattr(dataset, 'train'):
        formatted_data = dataset.train.map(format_example)
    elif isinstance(dataset, dict) and 'train' in dataset:
        formatted_data = dataset['train'].map(format_example)
    else:
        formatted_data = dataset.map(format_example)
    
    return formatted_data

def analyze_dataset(dataset, input_col, output_col):
    """Analyze the dataset for potential issues."""
    issues = {
        "empty_inputs": 0,
        "empty_outputs": 0,
        "very_short_inputs": 0,
        "very_short_outputs": 0,
        "very_long_inputs": 0,
        "very_long_outputs": 0,
        "total_examples": len(dataset)
    }
    
    # Create a copy of the keys to prevent dictionary size change during iteration
    examples = list(dataset)
    
    for example in examples:
        input_text = example.get(input_col, "").strip()
        output_text = example.get(output_col, "").strip()
        
        # Check for empty inputs/outputs
        if not input_text:
            issues["empty_inputs"] += 1
        if not output_text:
            issues["empty_outputs"] += 1
        
        # Check for very short inputs/outputs (less than 5 words)
        if len(input_text.split()) < 5:
            issues["very_short_inputs"] += 1
        if len(output_text.split()) < 5:
            issues["very_short_outputs"] += 1
        
        # Check for very long inputs/outputs (more than 500 words)
        if len(input_text.split()) > 500:
            issues["very_long_inputs"] += 1
        if len(output_text.split()) > 500:
            issues["very_long_outputs"] += 1
    
    # Calculate percentages
    total = issues["total_examples"]
    for key in list(issues.keys()):  # Use list() to create a copy of keys
        if key != "total_examples":
            issues[f"{key}_percent"] = (issues[key] / total) * 100
    
    return issues

def validate_formatted_data(dataset, separator=" ### "):
    """Validate that formatted data contains the separator."""
    if "text" not in dataset.column_names:
        logger.error("Formatted dataset does not have a 'text' column")
        return False
    
    separator_count = 0
    for example in dataset:
        if separator in example["text"]:
            separator_count += 1
    
    separator_percent = (separator_count / len(dataset)) * 100
    logger.info(f"{separator_count} out of {len(dataset)} examples ({separator_percent:.2f}%) have the separator")
    
    if separator_percent < 99:
        logger.warning(f"Only {separator_percent:.2f}% of examples have the separator. This may cause issues.")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Format training data with proper separators to prevent empty responses in LLM fine-tuning")
    parser.add_argument("--input", type=str, required=True, help="Path to input data file (json, jsonl, csv, arrow)")
    parser.add_argument("--output", type=str, required=True, help="Path to output data file")
    parser.add_argument("--separator", type=str, default=" ### ", help="Separator between input and output (default: ' ### ')")
    parser.add_argument("--input-col", type=str, help="Input column name (auto-detect if not provided)")
    parser.add_argument("--output-col", type=str, help="Output column name (auto-detect if not provided)")
    parser.add_argument("--format", type=str, choices=["text", "combined"], default="text", 
                        help="Output format: 'text' for a single text field or 'combined' for separate fields plus text")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    try:
        # Load data
        logger.info(f"Loading data from {args.input}")
        dataset = load_dataset('arrow', data_files=args.input)
        logger.info(f"Loaded {len(dataset['train'])} examples")
        
        # Detect or use provided column names
        input_col = args.input_col
        output_col = args.output_col
        
        if input_col is None or output_col is None:
            detected_input_col, detected_output_col = detect_column_types(dataset)
            input_col = args.input_col or detected_input_col
            output_col = args.output_col or detected_output_col
        
        # Analyze original dataset
        logger.info("Analyzing original dataset...")
        # Assuming train split exists
        issues = analyze_dataset(dataset['train'], input_col, output_col)
        
        # Report issues
        logger.info("\n=== DATASET ANALYSIS ===")
        logger.info(f"Total examples: {issues['total_examples']}")
        logger.info(f"Empty inputs: {issues['empty_inputs']} ({issues['empty_inputs_percent']:.2f}%)")
        logger.info(f"Empty outputs: {issues['empty_outputs']} ({issues['empty_outputs_percent']:.2f}%)")
        logger.info(f"Very short inputs (<5 words): {issues['very_short_inputs']} ({issues['very_short_inputs_percent']:.2f}%)")
        logger.info(f"Very short outputs (<5 words): {issues['very_short_outputs']} ({issues['very_short_outputs_percent']:.2f}%)")
        logger.info(f"Very long inputs (>500 words): {issues['very_long_inputs']} ({issues['very_long_inputs_percent']:.2f}%)")
        logger.info(f"Very long outputs (>500 words): {issues['very_long_outputs']} ({issues['very_long_outputs_percent']:.2f}%)")
        
        # Format data with separator
        logger.info(f"Formatting data with separator: '{args.separator}'")
        formatted_dataset = format_with_separator(dataset, input_col, output_col, args.separator, args.format)
        
        # Validate formatted data
        validate_formatted_data(formatted_dataset, args.separator)
        
        # Show some examples
        logger.info("\n=== SAMPLE FORMATTED EXAMPLES ===")
        for i in range(min(3, len(formatted_dataset))):
            sample = formatted_dataset[i]["text"]
            # Truncate if too long
            if len(sample) > 200:
                sample = sample[:197] + "..."
            logger.info(f"Example {i+1}: {sample}")
        
        # Save formatted data
        logger.info(f"Saving formatted data to {args.output}")
        
        if args.output.endswith('.arrow'):
            formatted_dataset.save_to_disk(args.output)
        elif args.output.endswith('.csv'):
            formatted_dataset.to_csv(args.output)
        elif args.output.endswith('.json') or args.output.endswith('.jsonl'):
            formatted_dataset.to_json(args.output)
        else:
            # Default to parquet
            formatted_dataset.to_parquet(args.output)
        
        logger.info(f"Successfully saved {len(formatted_dataset)} formatted examples")
        
    except Exception as e:
        logger.error(f"Error processing data: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
