"""
Enhanced Model Benchmark with SystemPromptInjector and RTX 3090 Optimizations
============================================================================
Production-ready benchmark with hardware optimizations and comprehensive evaluation.
"""

import time
import torch
import logging
import gc
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import json
import psutil
try:
    import GPUtil
except ImportError:
    GPUtil = None

# Import our custom modules
from .system_prompt_injector import SystemPromptInjector
from .enhanced_evaluation import EnhancedEvaluator, EvaluationResults

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs"""
    enable_flash_attention: bool = True
    enable_torch_compile: bool = True
    use_system_prompts: bool = True
    max_new_tokens: int = 512
    temperature: float = 0.7
    batch_size: int = 1
    num_runs: int = 3
    warmup_runs: int = 1

class RTXOptimizer:
    """RTX 3090 specific optimizations"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_memory_gb = self._get_gpu_memory()
        logger.info(f"GPU Memory: {self.gpu_memory_gb:.1f} GB")
    
    def _get_gpu_memory(self) -> float:
        """Get GPU memory in GB"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024**3
        return 0.0
    
    def optimize_model(self, model, config: BenchmarkConfig):
        """Apply RTX 3090 optimizations"""
        optimized_model = model
        
        # Enable Flash Attention 2 if available and requested
        if config.enable_flash_attention and hasattr(model, 'config'):
            try:
                if hasattr(model.config, 'use_flash_attention_2'):
                    model.config.use_flash_attention_2 = True
                    logger.info("✅ FlashAttention-2 enabled")
                else:
                    logger.info("ℹ️  FlashAttention-2 not available for this model")
            except Exception as e:
                logger.warning(f"FlashAttention-2 setup failed: {e}")
        
        # Apply torch.compile optimization
        if config.enable_torch_compile:
            try:
                # Check PyTorch version
                torch_version = torch.__version__.split('.')
                if int(torch_version[0]) >= 2:
                    optimized_model = torch.compile(model, mode="reduce-overhead")
                    logger.info("✅ torch.compile optimization enabled")
                else:
                    logger.info("ℹ️  torch.compile requires PyTorch 2.0+")
            except Exception as e:
                logger.warning(f"torch.compile optimization failed: {e}")
                optimized_model = model
        
        # Memory optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Set memory fraction for RTX 3090 (24GB)
            if self.gpu_memory_gb > 20:
                torch.cuda.set_per_process_memory_fraction(0.9)
            
        return optimized_model
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system performance information"""
        info = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "gpu_available": torch.cuda.is_available()
        }
        
        if torch.cuda.is_available() and GPUtil:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    info.update({
                        "gpu_memory_used": gpu.memoryUsed,
                        "gpu_memory_total": gpu.memoryTotal,
                        "gpu_memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                        "gpu_temperature": gpu.temperature,
                        "gpu_load": gpu.load * 100
                    })
            except Exception as e:
                logger.warning(f"GPU monitoring failed: {e}")
        
        return info

class EnhancedModelBenchmark:
    """Enhanced model benchmark with comprehensive evaluation"""
    
    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        self.prompt_injector = SystemPromptInjector()
        self.evaluator = EnhancedEvaluator()
        self.optimizer = RTXOptimizer()
        
        # Test datasets
        self.test_datasets = {
            "coding": [
                {
                    "prompt": "Write a Python function to find the longest palindromic substring in a string.",
                    "reference": "def longest_palindrome(s): # Implementation with proper algorithm",
                    "task_type": "coding"
                },
                {
                    "prompt": "Create a binary search tree class with insert, search, and delete methods.",
                    "reference": "class BST: def __init__(self): # Complete BST implementation",
                    "task_type": "coding"
                },
                {
                    "prompt": "Implement a decorator that measures function execution time.",
                    "reference": "import time; def timer(func): # Decorator implementation",
                    "task_type": "coding"
                }
            ],
            "reasoning": [
                {
                    "prompt": "A train travels 120 km in 2 hours, then 80 km in 1 hour. What's the average speed?",
                    "reference": "Average speed = total distance / total time = 200km / 3h = 66.67 km/h",
                    "task_type": "reasoning"
                },
                {
                    "prompt": "If 5 machines can produce 5 widgets in 5 minutes, how long for 100 machines to produce 100 widgets?",
                    "reference": "5 minutes. Each machine produces 1 widget in 5 minutes.",
                    "task_type": "reasoning"
                }
            ],
            "knowledge": [
                {
                    "prompt": "Explain the difference between supervised and unsupervised machine learning.",
                    "reference": "Supervised learning uses labeled data, unsupervised learning finds patterns in unlabeled data.",
                    "task_type": "knowledge"
                },
                {
                    "prompt": "What is the greenhouse effect and how does it work?",
                    "reference": "Greenhouse effect traps heat in atmosphere through greenhouse gases absorbing infrared radiation.",
                    "task_type": "knowledge"
                }
            ]
        }
        
        logger.info("EnhancedModelBenchmark initialized")
    
    def benchmark_model(self, model, tokenizer, model_name: str) -> Dict[str, Any]:
        """Run comprehensive benchmark on a model"""
        logger.info(f"🚀 Starting benchmark for {model_name}")
        
        # Optimize model for RTX 3090
        optimized_model = self.optimizer.optimize_model(model, self.config)
        
        results = {
            "model_name": model_name,
            "config": self.config.__dict__,
            "system_info": self.optimizer.get_system_info(),
            "task_results": {},
            "overall_metrics": {}
        }
        
        all_evaluations = []
        
        # Run tests for each task type
        for task_type, test_cases in self.test_datasets.items():
            logger.info(f"📊 Testing {task_type} tasks...")
            task_results = []
            
            for i, test_case in enumerate(test_cases):
                logger.info(f"  Test {i+1}/{len(test_cases)}: {test_case['prompt'][:50]}...")
                
                # Run multiple iterations for stability
                run_results = []
                for run in range(self.config.num_runs):
                    eval_result = self._run_single_test(
                        optimized_model, tokenizer, test_case, run == 0
                    )
                    run_results.append(eval_result)
                
                # Average the results
                avg_result = self._average_results(run_results)
                task_results.append(avg_result.__dict__)
                all_evaluations.append(avg_result)
            
            results["task_results"][task_type] = task_results
        
        # Calculate overall metrics
        results["overall_metrics"] = self._calculate_overall_metrics(all_evaluations)
        
        logger.info(f"✅ Benchmark completed for {model_name}")
        return results
    
    def _run_single_test(self, model, tokenizer, test_case: Dict, is_warmup: bool = False) -> EvaluationResults:
        """Run a single test case"""
        prompt = test_case["prompt"]
        reference = test_case["reference"]
        task_type = test_case["task_type"]
        
        # Apply system prompt if enabled
        if self.config.use_system_prompts:
            formatted_prompt = self.prompt_injector.format_prompt_with_system(prompt, task_type)
        else:
            formatted_prompt = prompt
        
        try:
            # Tokenize input
            inputs = tokenizer(
                formatted_prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            
            # Move to device if GPU available
            if torch.cuda.is_available():
                inputs = {k: v.to(self.optimizer.device) for k, v in inputs.items()}
            
            # Warmup run (not counted)
            if is_warmup:
                with torch.no_grad():
                    _ = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else tokenizer.pad_token_id
                    )
            
            # Timed generation
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else tokenizer.pad_token_id
                )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            # Decode response
            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            response_time = end_time - start_time
            
        except Exception as e:
            logger.error(f"Error during model generation: {e}")
            response = "Error generating response"
            response_time = 0.0
        
        # Analyze autonomy
        autonomy_metrics = self.prompt_injector.analyze_prompt_autonomy(response)
        
        # Comprehensive evaluation
        evaluation = self.evaluator.evaluate_comprehensive(
            response=response,
            reference=reference,
            response_time=response_time,
            task_type=task_type,
            autonomy_metrics=autonomy_metrics
        )
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return evaluation
    
    def _average_results(self, results: List[EvaluationResults]) -> EvaluationResults:
        """Average multiple evaluation results"""
        if not results:
            return None
        
        # Use first result as template
        avg_result = results[0]
        
        # Average numerical fields
        numerical_fields = [
            'rouge_l', 'response_time', 'autonomy_score', 'code_quality',
            'reasoning_quality', 'knowledge_accuracy', 'overall_score'
        ]
        
        for field in numerical_fields:
            values = [getattr(r, field) for r in results]
            setattr(avg_result, field, np.mean(values))
        
        return avg_result
    
    def _calculate_overall_metrics(self, evaluations: List[EvaluationResults]) -> Dict[str, float]:
        """Calculate overall performance metrics"""
        if not evaluations:
            return {}
        
        return {
            "avg_overall_score": np.mean([e.overall_score for e in evaluations]),
            "avg_response_time": np.mean([e.response_time for e in evaluations]),
            "avg_autonomy_score": np.mean([e.autonomy_score for e in evaluations]),
            "avg_rouge_l": np.mean([e.rouge_l for e in evaluations]),
            "avg_code_quality": np.mean([e.code_quality for e in evaluations]),
            "avg_reasoning_quality": np.mean([e.reasoning_quality for e in evaluations]),
            "avg_knowledge_accuracy": np.mean([e.knowledge_accuracy for e in evaluations]),
            "total_tests": len(evaluations),
            "autonomy_improvement": np.mean([e.autonomy_score for e in evaluations]) - 0.5  # Baseline comparison
        }
    
    def compare_models(self, models_config: List[Dict]) -> Dict[str, Any]:
        """Compare multiple models"""
        comparison_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "models": {},
            "comparison": {}
        }
        
        # Benchmark each model
        for model_config in models_config:
            model_name = model_config["name"]
            model = model_config["model"]
            tokenizer = model_config["tokenizer"]
            
            results = self.benchmark_model(model, tokenizer, model_name)
            comparison_results["models"][model_name] = results
        
        # Generate comparison metrics
        comparison_results["comparison"] = self._generate_comparison(comparison_results["models"])
        
        return comparison_results
    
    def _generate_comparison(self, models_results: Dict) -> Dict[str, Any]:
        """Generate model comparison insights"""
        if len(models_results) < 2:
            return {}
        
        model_names = list(models_results.keys())
        comparison = {}
        
        # Compare overall scores
        scores = {name: results["overall_metrics"]["avg_overall_score"] 
                 for name, results in models_results.items()}
        
        best_model = max(scores, key=scores.get)
        
        comparison["performance_ranking"] = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        comparison["best_overall"] = best_model
        comparison["score_differences"] = {
            name: scores[best_model] - score for name, score in scores.items()
        }
        
        # Speed comparison
        speeds = {name: results["overall_metrics"]["avg_response_time"] 
                 for name, results in models_results.items()}
        fastest_model = min(speeds, key=speeds.get)
        comparison["fastest_model"] = fastest_model
        
        # Autonomy comparison
        autonomy = {name: results["overall_metrics"]["avg_autonomy_score"] 
                   for name, results in models_results.items()}
        most_autonomous = max(autonomy, key=autonomy.get)
        comparison["most_autonomous"] = most_autonomous
        
        return comparison
    
    def save_results(self, results: Dict, filepath: str):
        """Save benchmark results to file"""
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        converted_results = convert_numpy(results)
        
        with open(filepath, 'w') as f:
            json.dump(converted_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")

# Quick test function
def test_enhanced_benchmark():
    """Test the enhanced benchmark system"""
    print("🧪 Testing Enhanced Model Benchmark")
    print("=" * 50)
    
    # Create test configuration
    config = BenchmarkConfig(
        enable_flash_attention=False,  # Set to False for testing without actual models
        enable_torch_compile=False,
        use_system_prompts=True,
        num_runs=1,
        max_new_tokens=100
    )
    
    benchmark = EnhancedModelBenchmark(config)
    
    # Test system prompt integration
    test_prompt = "Write a Python function to reverse a string"
    formatted = benchmark.prompt_injector.format_prompt_with_system(test_prompt)
    print(f"✅ System prompt integration working")
    print(f"   Original: {test_prompt}")
    print(f"   Formatted length: {len(formatted)} chars")
    
    # Test evaluation system
    sample_response = """
    Here's a Python function to reverse a string:
    
    ```python
    def reverse_string(s):
        return s[::-1]
    
    # Example usage
    result = reverse_string("hello")
    print(result)  # Output: "olleh"
    ```
    
    This function uses Python's slice notation with a step of -1 to reverse the string.
    """
    
    evaluation = benchmark.evaluator.evaluate_comprehensive(
        response=sample_response,
        reference="def reverse_string(s): return s[::-1]",
        response_time=0.5,
        task_type="coding",
        autonomy_metrics={"autonomy_score": 0.9, "question_count": 0}
    )
    
    print(f"✅ Enhanced evaluation working")
    print(f"   Overall Score: {evaluation.overall_score:.3f}")
    print(f"   Code Quality: {evaluation.code_quality:.3f}")
    print(f"   Autonomy Score: {evaluation.autonomy_score:.3f}")
    
    print("\n🚀 Enhanced benchmark system ready for production use!")

if __name__ == "__main__":
    test_enhanced_benchmark()
