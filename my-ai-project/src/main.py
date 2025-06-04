# main.py

import json
from src.task_config import load_model_comparison_results

def main():
    # Load model comparison results
    model_comparison_results = load_model_comparison_results('config/model_comparison_results.json')
    
    # Initialize AI model and execute tasks based on the configuration
    for task in model_comparison_results:
        prompt = task['prompt']
        # Here you would typically call your AI model with the prompt
        print(f"Executing task: {prompt}")

if __name__ == "__main__":
    main()