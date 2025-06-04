"""
Comprehensive Validation Script for Phi-2 Fine-tuned Model

This script provides thorough validation beyond loss metrics to ensure
your fine-tuned model is producing quality outputs and not empty responses.
"""

import os
import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import logging
from tqdm import tqdm
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("phi2_validation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
BASE_MODEL_ID = "microsoft/phi-2"
ADAPTER_PATH = "./models/fine-tuned/phi2-optimized"  # Path to your fine-tuned adapter
OUTPUT_DIR = "./validation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model_and_tokenizer(adapter_path=None):
    """Load the base or fine-tuned model."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    if adapter_path and os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        logger.info(f"Loading fine-tuned model from adapter: {adapter_path}")
        # Load the base model for merging with adapter
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Load and apply LoRA adapter
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        logger.info(f"Loading base model: {BASE_MODEL_ID}")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    
    return model, tokenizer

def load_test_examples(path=None):
    """Load test examples or use default ones if not provided."""
    if path and os.path.exists(path):
        logger.info(f"Loading test examples from: {path}")
        with open(path, "r") as f:
            test_examples = json.load(f)
    else:
        logger.info("Using default test examples")
        test_examples = [
            {
                "category": "Coding",
                "prompt": "Write a Python function to reverse a string while preserving the position of special characters.",
                "expected_pattern": "def"
            },
            {
                "category": "Knowledge",
                "prompt": "Explain the difference between supervised and unsupervised learning in machine learning.",
                "expected_pattern": "supervised|unsupervised|label|cluster"
            },
            {
                "category": "Problem Solving",
                "prompt": "How would you debug a memory leak in a Node.js application?",
                "expected_pattern": "heap|profile|memory|leak"
            },
            {
                "category": "Creative",
                "prompt": "Write a short story about a sentient AI that discovers emotions.",
                "expected_pattern": "feel|emotion|discover|learn"
            },
            {
                "category": "Empty Response Test",
                "prompt": "Create a comprehensive guide to setting up a secure AWS environment.",
                "expected_pattern": "AWS|security|cloud|VPC"
            }
        ]
    return test_examples

def evaluate_diversity(outputs):
    """Calculate diversity metrics among multiple outputs for the same prompt."""
    if len(outputs) < 2:
        return 0
    
    # Use BLEU score to measure similarity (lower means more diverse)
    bleu_scores = []
    for i, output1 in enumerate(outputs):
        for j, output2 in enumerate(outputs):
            if i < j:  # Only compare unique pairs
                # Convert to tokens
                ref = [output1.split()]
                hyp = output2.split()
                try:
                    bleu = sentence_bleu(ref, hyp)
                    bleu_scores.append(bleu)
                except Exception as e:
                    logger.warning(f"Error calculating BLEU score: {e}")
    
    # Diversity = 1 - average similarity
    if bleu_scores:
        diversity = 1 - np.mean(bleu_scores)
        return diversity
    else:
        return 0

def generate_response(model, tokenizer, prompt, temperature=0.7, max_length=512):
    """Generate a response from the model for a given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate with sampling for diversity
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Check if response is just repeating the prompt
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    
    return response

def calculate_metrics(original_response, expected_pattern):
    """Calculate metrics for a response."""
    # Check if response is empty or too short
    is_empty = len(original_response.strip()) == 0
    is_too_short = len(original_response.split()) < 5
    
    # Check if response contains expected pattern
    contains_pattern = expected_pattern.lower() in original_response.lower()
    
    # Calculate response length in tokens
    token_length = len(original_response.split())
    
    return {
        "is_empty": is_empty,
        "is_too_short": is_too_short,
        "contains_pattern": contains_pattern,
        "token_length": token_length
    }

def run_validation():
    """Run comprehensive validation on the model."""
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(ADAPTER_PATH)
    
    # Load test examples
    test_examples = load_test_examples()
    
    # Prepare results storage
    results = []
    all_responses = []
    
    # Run tests
    logger.info("Starting validation tests...")
    for example in tqdm(test_examples, desc="Testing examples"):
        category = example["category"]
        prompt = example["prompt"]
        expected_pattern = example["expected_pattern"]
        
        logger.info(f"Testing category: {category}")
        logger.info(f"Prompt: {prompt}")
        
        # Generate multiple responses to test consistency and diversity
        responses = []
        for i in range(3):  # Generate 3 responses per prompt
            try:
                response = generate_response(model, tokenizer, prompt)
                responses.append(response)
                
                # Calculate metrics
                metrics = calculate_metrics(response, expected_pattern)
                
                result = {
                    "category": category,
                    "prompt": prompt,
                    "response": response[:200] + "..." if len(response) > 200 else response,  # Truncate for display
                    "full_response": response,  # Keep the full response for detailed analysis
                    "sample_num": i + 1,
                    **metrics
                }
                
                results.append(result)
                logger.info(f"Sample {i+1} - Empty: {metrics['is_empty']} | Too short: {metrics['is_too_short']} | Contains pattern: {metrics['contains_pattern']} | Length: {metrics['token_length']}")
                
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                results.append({
                    "category": category,
                    "prompt": prompt,
                    "response": "ERROR",
                    "full_response": f"Error: {str(e)}",
                    "sample_num": i + 1,
                    "is_empty": True,
                    "is_too_short": True,
                    "contains_pattern": False,
                    "token_length": 0
                })
        
        # Calculate diversity for this prompt's responses
        diversity = evaluate_diversity(responses)
        for i in range(len(responses)):
            results[len(results) - 3 + i]["diversity"] = diversity
        
        logger.info(f"Response diversity: {diversity:.4f} (higher is more diverse)")
        all_responses.extend(responses)
    
    # Aggregate results
    df = pd.DataFrame(results)
    
    # Generate summary statistics
    empty_rate = df["is_empty"].mean() * 100
    too_short_rate = df["is_too_short"].mean() * 100
    pattern_match_rate = df["contains_pattern"].mean() * 100
    avg_length = df["token_length"].mean()
    avg_diversity = df["diversity"].mean()
    
    logger.info("\n=== VALIDATION SUMMARY ===")
    logger.info(f"Empty response rate: {empty_rate:.2f}%")
    logger.info(f"Too short response rate: {too_short_rate:.2f}%")
    logger.info(f"Expected pattern match rate: {pattern_match_rate:.2f}%")
    logger.info(f"Average response length: {avg_length:.2f} tokens")
    logger.info(f"Average response diversity: {avg_diversity:.4f}")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(OUTPUT_DIR, f"validation_results_{timestamp}.csv")
    df.to_csv(results_file, index=False)
    
    # Save summary
    summary = {
        "timestamp": timestamp,
        "model": ADAPTER_PATH if os.path.exists(ADAPTER_PATH) else BASE_MODEL_ID,
        "empty_rate": empty_rate,
        "too_short_rate": too_short_rate,
        "pattern_match_rate": pattern_match_rate,
        "avg_length": avg_length,
        "avg_diversity": avg_diversity,
        "num_examples": len(test_examples),
        "num_samples_per_example": 3
    }
    
    summary_file = os.path.join(OUTPUT_DIR, f"validation_summary_{timestamp}.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Detailed results saved to: {results_file}")
    logger.info(f"Summary saved to: {summary_file}")
    
    # Create visualizations
    create_visualizations(df, timestamp)
    
    return summary

def create_visualizations(df, timestamp):
    """Create visualizations of the validation results."""
    # Create output directory for plots
    plots_dir = os.path.join(OUTPUT_DIR, f"plots_{timestamp}")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot 1: Empty and too short response rates by category
    plt.figure(figsize=(10, 6))
    category_stats = df.groupby("category").agg({
        "is_empty": "mean", 
        "is_too_short": "mean",
        "contains_pattern": "mean"
    }) * 100
    
    ax = category_stats.plot(kind="bar", ylabel="Percentage (%)")
    plt.title("Response Quality by Category")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "quality_by_category.png"))
    
    # Plot 2: Response length distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df["token_length"], bins=20, alpha=0.7)
    plt.axvline(df["token_length"].mean(), color='r', linestyle='dashed', linewidth=1)
    plt.title("Response Length Distribution")
    plt.xlabel("Length (tokens)")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(plots_dir, "length_distribution.png"))
    
    # Plot 3: Diversity by category
    plt.figure(figsize=(10, 6))
    diversity_by_cat = df.groupby("category")["diversity"].mean()
    diversity_by_cat.plot(kind="bar")
    plt.title("Response Diversity by Category")
    plt.ylabel("Diversity Score (higher is better)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "diversity_by_category.png"))
    
    logger.info(f"Visualizations saved to: {plots_dir}")

if __name__ == "__main__":
    run_validation()
