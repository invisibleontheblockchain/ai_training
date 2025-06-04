"""
Task: Run Model Comparison

This task performs the following steps:
1. Load model outputs from the JSON file (located at c:\AI_Training\model_comparison_results.json).
2. Compare the responses of both the base and finetuned models using heuristic evaluations:
   - Code Completeness
   - Error Handling
   - Documentation Quality
   - Security Awareness
3. Print a side-by-side comparison with response time and a brief excerpt of the responses.
"""

import os
import subprocess

def run_model_comparison():
    # Set the path of the comparison script that processes the JSON file.
    script_path = os.path.join("c:\\", "AI_Training", "compare_models.py")
    if not os.path.exists(script_path):
        print(f"Comparison script not found at {script_path}")
        return

    # Run the comparison script using the integrated terminal command.
    subprocess.run(["python", script_path], shell=True)

if __name__ == "__main__":
    subprocess.run(["python", "c:\\AI_Training\\src\\task_config.py"], shell=True)
    run_model_comparison()