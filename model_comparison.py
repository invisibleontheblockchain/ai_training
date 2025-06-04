import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time
from typing import List, Dict
import subprocess
import re

class ModelComparison:
    def __init__(self, base_model_name: str, finetuned_path: str):
        """Initialize both base and fine-tuned models for comparison"""
        print("Loading models for comparison...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        print("Loading base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load fine-tuned model
        print("Loading fine-tuned model...")
        base_for_peft = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.finetuned_model = PeftModel.from_pretrained(
            base_for_peft, 
            finetuned_path
        )
        self.finetuned_model = self.finetuned_model.merge_and_unload()
        
        # Set to eval mode
        self.base_model.eval()
        self.finetuned_model.eval()
        
    def generate_response(self, model, prompt: str, max_length: int = 200) -> Dict:
        """Generate response from a model with timing"""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        generation_time = time.time() - start_time
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the response
        response = response[len(prompt):].strip()
        
        return {
            "response": response,
            "time": generation_time,
            "tokens": len(outputs[0]) - len(inputs['input_ids'][0])
        }
    
    def compare_responses(self, prompts: List[str], max_length: int = 200):
        """Compare responses from both models side by side"""
        results = []
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n{'='*80}")
            print(f"Test {i}: {prompt}")
            print('='*80)
            
            # Get base model response
            print("\n🔵 BASE MODEL:")
            base_result = self.generate_response(self.base_model, prompt, max_length)
            print(base_result['response'])
            print(f"(Generated in {base_result['time']:.2f}s, {base_result['tokens']} tokens)")
            
            # Get fine-tuned model response
            print("\n🟢 FINE-TUNED MODEL:")
            ft_result = self.generate_response(self.finetuned_model, prompt, max_length)
            print(ft_result['response'])
            print(f"(Generated in {ft_result['time']:.2f}s, {ft_result['tokens']} tokens)")
            
            results.append({
                "prompt": prompt,
                "base": base_result,
                "finetuned": ft_result
            })
        
        return results

def run_model(model, prompt):
    """Run a model with a prompt and return the output and time taken."""
    start = time.time()
    proc = subprocess.Popen(
        model["command"] + [prompt],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    try:
        out, err = proc.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()
        return "", 120.0, False
    elapsed = time.time() - start
    return out.strip(), elapsed, err

# Example usage
if __name__ == "__main__":
    # Initialize comparison
    comparison = ModelComparison(
        base_model_name="microsoft/phi-2",
        finetuned_path="C:/AI_Training/models/fine-tuned/cline-optimal-ghosteam"
    )
    
    # Test prompts - adjust these based on what you fine-tuned for
    test_prompts = [
        # General coding tasks
        "Write a Python function to calculate the factorial of a number:",
        
        # Specific to your training data
        "Explain how to implement error handling in Python:",
        
        # Code completion
        "def merge_sort(arr):\n    # Implementation of merge sort algorithm",
        
        # Debugging
        "Debug this code:\ndef add_numbers(a, b):\n    return a + b\nresult = add_numbers(5)",
        
        # Best practices
        "What are the best practices for writing clean Python code?",
        
        # Your specific use case (adjust based on your training data)
        "How do I optimize this function for better performance?"
    ]
    
    # Run comparison
    results = comparison.compare_responses(test_prompts, max_length=150)
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print('='*80)
    
    base_times = [r['base']['time'] for r in results]
    ft_times = [r['finetuned']['time'] for r in results]
    
    print(f"Average generation time:")
    print(f"  Base model: {sum(base_times)/len(base_times):.2f}s")
    print(f"  Fine-tuned: {sum(ft_times)/len(ft_times):.2f}s")
    
    # Save results for further analysis
    import json
    with open("model_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to model_comparison_results.json")
    
    # Additional script to compare cline-optimal:latest and the base model using several prompts
    print("\n=== Additional Model Comparison Script ===")
    # Models to compare (update these as needed)
    MODELS = [
        {
            "name": "cline-optimal:latest",
            "command": ["ollama", "run", "cline-optimal:latest"],
        },
        {
            "name": "qwen2.5-coder:7b-instruct-q4_K_M (base)",
            "command": ["ollama", "run", "qwen2.5-coder:7b-instruct-q4_K_M"],
        },
    ]

    PROMPTS = [
        "Write a Python function to check if a string is a palindrome.",
        "Design a microservice architecture for an e-commerce platform.",
        "Debug this code: for i in range(10): print(i) - it's not working.",
        "Create a simple Flask API for user authentication.",
    ]

    EXPECTED_PATTERNS = [
        "def is_palindrome",
        "service|component|API|database",
        "indent|syntax|error|fix",
        "from flask import|@app.route",
    ]

    results = []
    for prompt_idx, prompt in enumerate(PROMPTS):
        print(f"\n=== Prompt {prompt_idx+1}: {prompt}")
        for model in MODELS:
            print(f"\nModel: {model['name']}")
            output, elapsed, err = run_model(model, prompt)
            is_empty = len(output.strip()) == 0
            pattern = EXPECTED_PATTERNS[prompt_idx]
            success = bool(re.search(pattern, output, re.IGNORECASE))
            print(f"Time: {elapsed:.2f}s | Success: {success} | Empty: {is_empty}")
            print(f"Output (first 300 chars):\n{output[:300]}")
            if err:
                print(f"[stderr]: {err}")
            results.append({
                "model": model["name"],
                "prompt": prompt,
                "success": success,
                "time": elapsed,
                "empty": is_empty,
                "output": output[:300],
            })
    # Summary
    print("\n=== SUMMARY ===")
    for model in MODELS:
        model_results = [r for r in results if r["model"] == model["name"]]
        success_rate = sum(r["success"] for r in model_results) / len(model_results) * 100
        avg_time = sum(r["time"] for r in model_results) / len(model_results)
        print(f"{model['name']}: Success Rate: {success_rate:.1f}% | Avg Time: {avg_time:.2f}s")