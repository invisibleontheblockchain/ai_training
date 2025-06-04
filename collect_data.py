
import os
import json
from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetDict
from tqdm import tqdm

class SimpleDataCollector:
    """Simplified data collector for GhosTeam"""
    
    def __init__(self, ghosteam_path, output_path):
        self.ghosteam_path = Path(ghosteam_path)
        self.output_path = Path(output_path)
        
    def collect_python_files(self):
        """Collect all Python files from GhosTeam"""
        print("Collecting Python files...")
        data = []
        
        # Try the provided path first
        if not self.ghosteam_path.exists():
            # Try alternative path without 'ghosteam' at the end
            alt_path = self.ghosteam_path.parent
            if alt_path.exists():
                print(f"Using alternative path: {alt_path}")
                self.ghosteam_path = alt_path
        
        for py_file in self.ghosteam_path.rglob("*.py"):
            if "__pycache__" in str(py_file) or ".venv" in str(py_file):
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8')
                
                # Simple function extraction
                lines = content.split('\n')
                current_function = []
                in_function = False
                indent_level = 0
                
                for i, line in enumerate(lines):
                    if line.strip().startswith('def '):
                        if current_function:
                            # Save previous function
                            func_content = '\n'.join(current_function)
                            func_name = current_function[0].split('(')[0].replace('def ', '').strip()
                            data.append({
                                'instruction': f"Write a function named {func_name}",
                                'input': f"File: {py_file.name}",
                                'output': func_content
                            })
                        current_function = [line]
                        in_function = True
                        indent_level = len(line) - len(line.lstrip())
                    elif in_function:
                        # Check if we're still in the function
                        if line.strip() and not line[0].isspace():
                            # New top-level statement, function ended
                            if current_function:
                                func_content = '\n'.join(current_function)
                                func_name = current_function[0].split('(')[0].replace('def ', '').strip()
                                data.append({
                                    'instruction': f"Write a function named {func_name}",
                                    'input': f"File: {py_file.name}",
                                    'output': func_content
                                })
                            current_function = []
                            in_function = False
                        else:
                            current_function.append(line)
                
                # Don't forget the last function
                if current_function:
                    func_content = '\n'.join(current_function)
                    func_name = current_function[0].split('(')[0].replace('def ', '').strip()
                    data.append({
                        'instruction': f"Write a function named {func_name}",
                        'input': f"File: {py_file.name}",
                        'output': func_content
                    })
                        
            except Exception as e:
                print(f"Error processing {py_file}: {e}")
                
        print(f"Collected {len(data)} functions")
        return data
    
    def create_dataset(self):
        """Create training dataset"""
        data = self.collect_python_files()
        
        if not data:
            print("No data collected! Creating sample data...")
            # Create some sample data
            data = [
                {
                    'instruction': 'Write a hello world function',
                    'input': 'Create a simple greeting function',
                    'output': 'def hello_world():\n    print("Hello, World!")'
                },
                {
                    'instruction': 'Write a function to add two numbers',
                    'input': 'Create a function that adds two numbers',
                    'output': 'def add_numbers(a, b):\n    return a + b'
                }
            ]
            
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Split into train/validation
        train_size = int(0.9 * len(df))
        train_df = df[:train_size]
        val_df = df[train_size:] if train_size < len(df) else df[:1]  # Ensure we have at least one validation sample
        
        # Create dataset
        dataset_dict = DatasetDict({
            'train': Dataset.from_pandas(train_df),
            'validation': Dataset.from_pandas(val_df)
        })
        
        # Save dataset
        output_dir = self.output_path / "datasets" / "processed" / "ghosteam_training"
        dataset_dict.save_to_disk(str(output_dir))
        print(f"Dataset saved to {output_dir}")
        
        return dataset_dict

if __name__ == "__main__":
    collector = SimpleDataCollector(
        ghosteam_path="C:/Users/avion/OneDrive/Documents/GitHub/ghosteam",
        output_path="C:/AI_Training"
    )
    collector.create_dataset()
