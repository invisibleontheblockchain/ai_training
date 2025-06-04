import json
import os

def load_model_comparison_results(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    with open(file_path, 'r') as file:
        return json.load(file)

def get_task_configurations(results):
    task_configs = []
    for result in results:
        task_configs.append({
            "prompt": result["prompt"],
            "base_response": result["base"]["response"],
            "finetuned_response": result["finetuned"]["response"],
            "base_time": result["base"]["time"],
            "finetuned_time": result["finetuned"]["time"],
            "base_tokens": result["base"]["tokens"],
            "finetuned_tokens": result["finetuned"]["tokens"]
        })
    return task_configs

def main():
    file_path = os.path.join(os.path.dirname(__file__), '../config/model_comparison_results.json')
    results = load_model_comparison_results(file_path)
    task_configurations = get_task_configurations(results)
    return task_configurations

if __name__ == "__main__":
    main()