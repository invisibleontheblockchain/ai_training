"""
Model Comparison Benchmark - Phi-2 Fine-tuned vs Cline-Optimal
===================================================
This script compares performance between:
1. Fine-tuned Phi-2 model with QLoRA
2. Cline-optimal model

Metrics measured:
- Response time
- Response quality
- Token efficiency
- Resource usage
"""

import os
import time
import json
import torch
import psutil
import numpy as np
from datetime import datetime
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline,
    BitsAndBytesConfig
)
from peft import PeftModel, PeftConfig
import subprocess
from rouge_score import rouge_scorer
import GPUtil

# Configure paths
PHI2_FINETUNED_PATH = "c:/AI_Training/models/fine-tuned/phi2-optimized"
RESULTS_PATH = "c:/AI_Training/model_comparison_results.json"

# Test prompts across different categories
TEST_PROMPTS = {
    "coding": [
        "Write a Python function to check if a string is a palindrome.",
        "Create a simple REST API using Flask with two endpoints.",
        "Write a function to find the maximum depth of a binary tree.",
        "Implement a quicksort algorithm in Python.",
        "Create a class for handling file operations with error handling."
    ],
    "knowledge": [
        "Explain the difference between supervised and unsupervised learning.",
        "What is backpropagation and how does it work?",
        "Describe the transformer architecture in NLP.",
        "What is overfitting and how can it be prevented?",
        "Explain the concept of gradient descent in machine learning."
    ],
    "reasoning": [
        "A box contains 10 white balls, 8 black balls, and 6 red balls. What is the probability of drawing a white ball?",
        "If a train travels at 60 mph for 3 hours, and then 80 mph for 2 hours, what is the average speed?",
        "Solve the equation: 3x + 7 = 22",
        "If the sequence is 2, 6, 18, 54, what is the next number?",
        "A shopkeeper buys an item for $80 and sells it at a 25% profit. What is the selling price?"
    ]
}

class ModelBenchmark:
    def __init__(self):
        self.results = {
            "phi2_finetuned": {"responses": {}, "metrics": {}},
            "cline_optimal": {"responses": {}, "metrics": {}}
        }
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def setup_phi2_model(self):
        """Setup the fine-tuned Phi-2 model with QLoRA"""
        print("\n== Setting up Phi-2 Fine-tuned Model ==")
        start_time = time.time()
        
        # Check if model exists
        if not os.path.exists(PHI2_FINETUNED_PATH):
            print(f"Error: Fine-tuned model not found at {PHI2_FINETUNED_PATH}")
            return False
            
        # Setup quantization configuration
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        # Load base model and tokenizer
        try:
            # Get the base model from the config
            peft_config = PeftConfig.from_pretrained(PHI2_FINETUNED_PATH)
            base_model_name = peft_config.base_model_name_or_path
            
            print(f"Loading base model: {base_model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                trust_remote_code=True
            )
            
            # Load the LoRA adapter
            print(f"Loading LoRA adapter from: {PHI2_FINETUNED_PATH}")
            model = PeftModel.from_pretrained(base_model, PHI2_FINETUNED_PATH)
            
            # Create a text generation pipeline
            self.phi2_pipeline = pipeline(
                "text-generation", 
                model=model, 
                tokenizer=tokenizer
            )
            
            load_time = time.time() - start_time
            print(f"Phi-2 model loaded in {load_time:.2f} seconds")
            
            # Get model size
            model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            print(f"Model size: {model_size_mb:.2f} MB")
            
            self.results["phi2_finetuned"]["metrics"]["load_time"] = load_time
            self.results["phi2_finetuned"]["metrics"]["model_size_mb"] = model_size_mb
            
            return True
            
        except Exception as e:
            print(f"Error loading Phi-2 model: {str(e)}")
            return False
    
    def setup_cline_optimal(self):
        """Setup the cline-optimal model through Ollama"""
        print("\n== Setting up Cline-Optimal Model ==")
        start_time = time.time()
        
        # Check if Ollama is installed
        try:
            subprocess.run(["ollama", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: Ollama is not installed or not in PATH")
            return False
            
        # Check if cline-optimal model exists
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if "cline-optimal:latest" not in result.stdout:
            print("Error: cline-optimal model not found in Ollama")
            print("Please run the rtx3090_cline_optimal_system.ps1 script first")
            return False
            
        load_time = time.time() - start_time
        print(f"Cline-Optimal model ready in {load_time:.2f} seconds")
        
        self.results["cline_optimal"]["metrics"]["load_time"] = load_time
        return True
        
    def run_phi2_benchmarks(self):
        """Run benchmarks on the Phi-2 fine-tuned model"""
        print("\n== Running Phi-2 Fine-tuned Benchmarks ==")
        
        category_times = []
        category_token_counts = []
        
        for category, prompts in TEST_PROMPTS.items():
            print(f"\nCategory: {category}")
            for i, prompt in enumerate(prompts):
                print(f"  Prompt {i+1}/{len(prompts)}: {prompt[:40]}...")
                
                # Track GPU usage before generation
                gpu = GPUtil.getGPUs()[0]
                gpu_before = {
                    "memory_used": gpu.memoryUsed,
                    "load": gpu.load
                }
                
                # Generate text and measure time
                start_time = time.time()
                result = self.phi2_pipeline(
                    prompt, 
                    max_new_tokens=256, 
                    do_sample=True, 
                    temperature=0.7,
                    return_full_text=False
                )
                response_time = time.time() - start_time
                
                # Track GPU usage after generation
                gpu = GPUtil.getGPUs()[0]
                gpu_after = {
                    "memory_used": gpu.memoryUsed,
                    "load": gpu.load
                }
                
                response = result[0]['generated_text']
                
                # Calculate tokens per second
                input_tokens = len(self.phi2_pipeline.tokenizer.encode(prompt))
                output_tokens = len(self.phi2_pipeline.tokenizer.encode(response))
                tokens_per_second = output_tokens / response_time if response_time > 0 else 0
                
                # Store results
                prompt_key = f"{category}_{i+1}"
                self.results["phi2_finetuned"]["responses"][prompt_key] = {
                    "prompt": prompt,
                    "response": response,
                    "response_time": response_time,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "tokens_per_second": tokens_per_second,
                    "gpu_memory_used_mb": gpu_after["memory_used"] - gpu_before["memory_used"],
                    "gpu_load_change": gpu_after["load"] - gpu_before["load"]
                }
                
                category_times.append(response_time)
                category_token_counts.append(output_tokens)
                
                print(f"    Response time: {response_time:.2f}s, Tokens: {output_tokens}, Tokens/sec: {tokens_per_second:.2f}")
        
        # Calculate overall metrics
        self.results["phi2_finetuned"]["metrics"]["avg_response_time"] = np.mean(category_times)
        self.results["phi2_finetuned"]["metrics"]["avg_tokens_per_response"] = np.mean(category_token_counts)
        self.results["phi2_finetuned"]["metrics"]["avg_tokens_per_second"] = np.mean([r["tokens_per_second"] for r in self.results["phi2_finetuned"]["responses"].values()])
        
        print(f"\nPhi-2 Average response time: {self.results['phi2_finetuned']['metrics']['avg_response_time']:.2f}s")
        print(f"Phi-2 Average tokens per response: {self.results['phi2_finetuned']['metrics']['avg_tokens_per_response']:.2f}")
        print(f"Phi-2 Average tokens per second: {self.results['phi2_finetuned']['metrics']['avg_tokens_per_second']:.2f}")
    
    def run_cline_optimal_benchmarks(self):
        """Run benchmarks on the cline-optimal model"""
        print("\n== Running Cline-Optimal Benchmarks ==")
        
        import ollama  # Import here to avoid dependencies if only using Phi-2
        
        category_times = []
        category_token_counts = []
        
        for category, prompts in TEST_PROMPTS.items():
            print(f"\nCategory: {category}")
            for i, prompt in enumerate(prompts):
                print(f"  Prompt {i+1}/{len(prompts)}: {prompt[:40]}...")
                
                # Track resources before generation
                gpu = GPUtil.getGPUs()[0]
                gpu_before = {
                    "memory_used": gpu.memoryUsed,
                    "load": gpu.load
                }
                
                # Generate text and measure time
                start_time = time.time()
                result = ollama.chat(
                    model="cline-optimal:latest",
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.7}
                )
                response_time = time.time() - start_time
                
                # Track GPU usage after generation
                gpu = GPUtil.getGPUs()[0]
                gpu_after = {
                    "memory_used": gpu.memoryUsed,
                    "load": gpu.load
                }
                
                response = result["message"]["content"]
                
                # Estimate token count (rough approximation)
                # Ollama doesn't provide direct token counts, so we estimate
                input_tokens = len(prompt.split())
                output_tokens = len(response.split())
                tokens_per_second = output_tokens / response_time if response_time > 0 else 0
                
                # Store results
                prompt_key = f"{category}_{i+1}"
                self.results["cline_optimal"]["responses"][prompt_key] = {
                    "prompt": prompt,
                    "response": response,
                    "response_time": response_time,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "tokens_per_second": tokens_per_second,
                    "gpu_memory_used_mb": gpu_after["memory_used"] - gpu_before["memory_used"],
                    "gpu_load_change": gpu_after["load"] - gpu_before["load"]
                }
                
                category_times.append(response_time)
                category_token_counts.append(output_tokens)
                
                print(f"    Response time: {response_time:.2f}s, Tokens (est): {output_tokens}, Tokens/sec (est): {tokens_per_second:.2f}")
        
        # Calculate overall metrics
        self.results["cline_optimal"]["metrics"]["avg_response_time"] = np.mean(category_times)
        self.results["cline_optimal"]["metrics"]["avg_tokens_per_response"] = np.mean(category_token_counts)
        self.results["cline_optimal"]["metrics"]["avg_tokens_per_second"] = np.mean([r["tokens_per_second"] for r in self.results["cline_optimal"]["responses"].values()])
        
        print(f"\nCline-Optimal Average response time: {self.results['cline_optimal']['metrics']['avg_response_time']:.2f}s")
        print(f"Cline-Optimal Average tokens per response: {self.results['cline_optimal']['metrics']['avg_tokens_per_response']:.2f}")
        print(f"Cline-Optimal Average tokens per second: {self.results['cline_optimal']['metrics']['avg_tokens_per_second']:.2f}")
    
    def compare_responses(self):
        """Compare response quality between models"""
        print("\n== Comparing Response Quality ==")
        
        rouge_scores = []
        for key in self.results["phi2_finetuned"]["responses"].keys():
            phi2_response = self.results["phi2_finetuned"]["responses"][key]["response"]
            cline_response = self.results["cline_optimal"]["responses"][key]["response"]
            
            # Calculate ROUGE scores for similarity
            scores = self.rouge_scorer.score(phi2_response, cline_response)
            rouge_l = scores["rougeL"].fmeasure
            
            # Store comparison results
            self.results["comparison"] = self.results.get("comparison", {})
            self.results["comparison"][key] = {
                "rouge_l": rouge_l,
                "phi2_response_length": len(phi2_response),
                "cline_response_length": len(cline_response),
                "length_ratio": len(cline_response) / len(phi2_response) if len(phi2_response) > 0 else 0
            }
            
            rouge_scores.append(rouge_l)
            
            category, prompt_num = key.split("_")
            print(f"Prompt {category}_{prompt_num}: ROUGE-L: {rouge_l:.4f}, Length ratio: {self.results['comparison'][key]['length_ratio']:.2f}")
        
        # Calculate overall similarity
        self.results["comparison"]["metrics"] = {
            "avg_rouge_l": np.mean(rouge_scores),
            "similarity_percent": np.mean(rouge_scores) * 100
        }
        
        print(f"\nAverage similarity (ROUGE-L): {self.results['comparison']['metrics']['avg_rouge_l']:.4f} ({self.results['comparison']['metrics']['similarity_percent']:.2f}%)")
    
    def generate_summary(self):
        """Generate a summary of the benchmark results"""
        print("\n== Benchmark Summary ==")
        
        # Timing comparison
        phi2_time = self.results["phi2_finetuned"]["metrics"]["avg_response_time"]
        cline_time = self.results["cline_optimal"]["metrics"]["avg_response_time"]
        time_diff_percent = ((cline_time - phi2_time) / phi2_time) * 100
        
        # Token efficiency comparison
        phi2_tokens_per_sec = self.results["phi2_finetuned"]["metrics"]["avg_tokens_per_second"]
        cline_tokens_per_sec = self.results["cline_optimal"]["metrics"]["avg_tokens_per_second"]
        tokens_diff_percent = ((cline_tokens_per_sec - phi2_tokens_per_sec) / phi2_tokens_per_sec) * 100
        
        # Summary
        self.results["summary"] = {
            "benchmark_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "phi2_finetuned": {
                "avg_response_time": phi2_time,
                "avg_tokens_per_second": phi2_tokens_per_sec
            },
            "cline_optimal": {
                "avg_response_time": cline_time,
                "avg_tokens_per_second": cline_tokens_per_sec
            },
            "comparison": {
                "time_diff_percent": time_diff_percent,
                "tokens_diff_percent": tokens_diff_percent,
                "similarity_percent": self.results["comparison"]["metrics"]["similarity_percent"]
            }
        }
        
        # Display summary
        print(f"Benchmark date: {self.results['summary']['benchmark_date']}")
        print("\nPerformance Metrics:")
        print(f"  Phi-2 Fine-tuned avg response time: {phi2_time:.2f}s")
        print(f"  Cline-Optimal avg response time: {cline_time:.2f}s")
        print(f"  Time difference: {time_diff_percent:.2f}% ({'faster' if time_diff_percent < 0 else 'slower'} for Cline-Optimal)")
        
        print("\nToken Efficiency:")
        print(f"  Phi-2 Fine-tuned avg tokens/sec: {phi2_tokens_per_sec:.2f}")
        print(f"  Cline-Optimal avg tokens/sec: {cline_tokens_per_sec:.2f}")
        print(f"  Efficiency difference: {tokens_diff_percent:.2f}% ({'better' if tokens_diff_percent > 0 else 'worse'} for Cline-Optimal)")
        
        print("\nResponse Similarity:")
        print(f"  Average similarity: {self.results['comparison']['metrics']['similarity_percent']:.2f}%")
        
        # Overall assessment
        strengths = []
        if time_diff_percent < -10:  # Cline is at least 10% faster
            strengths.append("Cline-Optimal has significantly faster response times")
        elif time_diff_percent > 10:  # Phi2 is at least 10% faster
            strengths.append("Phi-2 Fine-tuned has significantly faster response times")
            
        if tokens_diff_percent > 10:  # Cline produces tokens faster
            strengths.append("Cline-Optimal has higher token efficiency")
        elif tokens_diff_percent < -10:  # Phi2 produces tokens faster
            strengths.append("Phi-2 Fine-tuned has higher token efficiency")
            
        if self.results["comparison"]["metrics"]["similarity_percent"] > 70:
            strengths.append("Models produce generally similar outputs")
        else:
            strengths.append("Models produce significantly different outputs")
            
        print("\nKey Observations:")
        for i, strength in enumerate(strengths):
            print(f"  {i+1}. {strength}")
    
    def save_results(self):
        """Save benchmark results to a file"""
        with open(RESULTS_PATH, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {RESULTS_PATH}")
    
    def run_all_benchmarks(self):
        """Run all benchmarks"""
        print("\n===============================================")
        print("   MODEL COMPARISON BENCHMARK - PHI2 vs CLINE  ")
        print("===============================================")
        
        # Setup models
        phi2_ready = self.setup_phi2_model()
        cline_ready = self.setup_cline_optimal()
        
        if not phi2_ready or not cline_ready:
            print("\nError: Could not set up one or both models. Benchmark aborted.")
            return
            
        # Run benchmarks
        self.run_phi2_benchmarks()
        self.run_cline_optimal_benchmarks()
        
        # Compare and summarize
        self.compare_responses()
        self.generate_summary()
        self.save_results()
        
        print("\n===============================================")
        print("   BENCHMARK COMPLETE                          ")
        print("===============================================")


if __name__ == "__main__":
    # Check if required libraries are installed
    try:
        import GPUtil
        import psutil
        from rouge_score import rouge_scorer
    except ImportError:
        print("Installing required packages...")
        subprocess.run(["pip", "install", "gputil", "psutil", "rouge-score"], check=True)
    
    benchmark = ModelBenchmark()
    benchmark.run_all_benchmarks()
