"""
Comprehensive Model Comparison Test Suite
==========================================
Compares your existing fine-tuned model against the new cline-optimal model
with multiple levels of testing including:

1. Speed Benchmarks
2. Quality Assessments  
3. Autonomy Testing
4. Task-Specific Evaluations
5. Resource Usage Analysis
6. Head-to-Head Comparison

Created: June 4, 2025
"""

import os
import time
import json
import torch
import psutil
import numpy as np
import ollama
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# Enhanced imports for comprehensive testing
try:
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        pipeline,
        BitsAndBytesConfig
    )
    from peft import PeftModel, PeftConfig
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️ Transformers/PEFT not available - local model testing will be limited")

try:
    import GPUtil
    GPU_MONITORING = True
except ImportError:
    GPU_MONITORING = False
    print("⚠️ GPUtil not available - GPU monitoring disabled")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("⚠️ ROUGE scoring not available - quality metrics will be limited")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_model_comparison.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TestConfig:
    """Configuration for comprehensive testing"""
    # Model paths
    phi2_finetuned_path: str = "c:/AI_Training/models/fine-tuned/phi2-optimized"
    cline_optimal_model: str = "cline-optimal:latest"
    
    # Testing parameters
    warmup_iterations: int = 2
    test_iterations: int = 5
    max_new_tokens: int = 512
    temperature: float = 0.7
    
    # Quality thresholds
    min_response_length: int = 10
    max_response_time: float = 30.0
    
    # Output configuration
    results_dir: str = "c:/AI_Training/comparison_results"
    save_detailed_logs: bool = True
    
@dataclass 
class ModelMetrics:
    """Metrics for a single model"""
    model_name: str
    avg_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    avg_tokens_per_second: float = 0.0
    avg_response_length: float = 0.0
    success_rate: float = 0.0
    memory_usage_mb: float = 0.0
    autonomy_score: float = 0.0
    code_quality_score: float = 0.0
    reasoning_quality_score: float = 0.0
    knowledge_accuracy_score: float = 0.0
    overall_quality_score: float = 0.0

class ComprehensiveModelComparison:
    """Comprehensive model comparison testing framework"""
    
    def __init__(self, config: TestConfig = None):
        self.config = config or TestConfig()
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config.__dict__,
            "models": {},
            "comparisons": {},
            "summary": {}
        }
        
        # Initialize scoring if available
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Create results directory
        Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)
        
        # Test suites for different complexity levels
        self.test_suites = self._create_test_suites()
        
        logger.info("Starting Comprehensive Model Comparison")
    
    def _create_test_suites(self) -> Dict[str, List[Dict]]:
        """Create test suites with different complexity levels"""
        return {
            "basic": [
                {
                    "prompt": "Hello, how are you?",
                    "category": "conversation",
                    "expected_keywords": ["hello", "good", "fine", "well"],
                    "min_length": 5,
                    "task_type": "general"
                },
                {
                    "prompt": "What is 2 + 2?",
                    "category": "simple_math",
                    "expected_keywords": ["4", "four"],
                    "min_length": 1,
                    "task_type": "reasoning"
                },
                {
                    "prompt": "Write a function to add two numbers",
                    "category": "simple_coding",
                    "expected_keywords": ["def", "function", "add", "+", "return"],
                    "min_length": 20,
                    "task_type": "coding"
                }
            ],
            "intermediate": [
                {
                    "prompt": "Explain the difference between a list and a tuple in Python",
                    "category": "technical_knowledge",
                    "expected_keywords": ["list", "tuple", "mutable", "immutable", "python"],
                    "min_length": 50,
                    "task_type": "knowledge"
                },
                {
                    "prompt": "Write a binary search algorithm in Python",
                    "category": "algorithm_coding",
                    "expected_keywords": ["def", "binary", "search", "array", "mid", "left", "right"],
                    "min_length": 100,
                    "task_type": "coding"
                },
                {
                    "prompt": "A train travels 120 km in 2 hours, then 80 km in 1 hour. What's the average speed?",
                    "category": "math_reasoning",
                    "expected_keywords": ["speed", "distance", "time", "200", "66"],
                    "min_length": 30,
                    "task_type": "reasoning"
                }
            ],
            "advanced": [
                {
                    "prompt": "Design a microservice architecture for an e-commerce platform with high availability requirements",
                    "category": "system_design",
                    "expected_keywords": ["microservice", "service", "database", "api", "load", "scaling"],
                    "min_length": 200,
                    "task_type": "reasoning"
                },
                {
                    "prompt": "Implement a thread-safe LRU cache in Python with O(1) operations",
                    "category": "advanced_coding",
                    "expected_keywords": ["class", "LRU", "cache", "thread", "lock", "dict", "OrderedDict"],
                    "min_length": 150,
                    "task_type": "coding"
                },
                {
                    "prompt": "Explain quantum computing principles and their advantages over classical computing",
                    "category": "advanced_knowledge",
                    "expected_keywords": ["quantum", "qubit", "superposition", "entanglement", "classical"],
                    "min_length": 150,
                    "task_type": "knowledge"
                }
            ],
            "autonomous": [
                {
                    "prompt": "Create a complete Flask API for user authentication without asking any questions",
                    "category": "autonomous_coding",
                    "expected_keywords": ["flask", "app", "route", "login", "auth", "user"],
                    "min_length": 200,
                    "task_type": "coding",
                    "test_autonomy": True
                },
                {
                    "prompt": "Set up a CI/CD pipeline for a Python project",
                    "category": "autonomous_devops",
                    "expected_keywords": ["pipeline", "CI", "CD", "github", "action", "test"],
                    "min_length": 150,
                    "task_type": "reasoning",
                    "test_autonomy": True
                }
            ]
        }
    
    def setup_phi2_model(self) -> bool:
        """Setup the fine-tuned Phi-2 model"""
        if not HF_AVAILABLE:
            logger.error("Transformers not available - cannot load Phi-2 model")
            return False
            
        logger.info("Setting up Phi-2 fine-tuned model...")
        
        try:
            # Check if model exists
            if not os.path.exists(self.config.phi2_finetuned_path):
                logger.error(f"Phi-2 model not found at {self.config.phi2_finetuned_path}")
                return False
            
            start_time = time.time()
            
            # Load PEFT config and base model
            peft_config = PeftConfig.from_pretrained(self.config.phi2_finetuned_path)
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            # Load fine-tuned model
            self.phi2_model = PeftModel.from_pretrained(base_model, self.config.phi2_finetuned_path)
            self.phi2_tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, trust_remote_code=True)
            
            # Create pipeline
            self.phi2_pipeline = pipeline(
                "text-generation",
                model=self.phi2_model,
                tokenizer=self.phi2_tokenizer,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            load_time = time.time() - start_time
            logger.info(f"OK: Phi-2 model loaded in {load_time:.2f} seconds")
            
            # Store setup metrics
            model_size_mb = sum(p.numel() * p.element_size() for p in self.phi2_model.parameters()) / (1024 * 1024)
            self.results["models"]["phi2_finetuned"] = {
                "setup_time": load_time,
                "model_size_mb": model_size_mb,
                "device": str(next(self.phi2_model.parameters()).device),
                "dtype": str(next(self.phi2_model.parameters()).dtype)
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up Phi-2 model: {str(e)}")
            return False
    
    def setup_cline_optimal(self) -> bool:
        """Setup the cline-optimal model through Ollama"""
        logger.info("Setting up cline-optimal model...")
        
        try:
            # Check if Ollama is available
            result = ollama.list()
            # Unpack first element of the tuple to get the list of models
            models = [model.model for model in result[0]]
            
            if self.config.cline_optimal_model not in models:
                logger.error(f"Model {self.config.cline_optimal_model} not found in Ollama")
                logger.info("Available models: " + ", ".join(models))
                return False
            
            # Test connectivity
            start_time = time.time()
            test_response = ollama.chat(
                model=self.config.cline_optimal_model,
                messages=[{"role": "user", "content": "test"}]
            )
            setup_time = time.time() - start_time
            
            logger.info(f"OK: Cline-optimal model ready in {setup_time:.2f} seconds")
            
            # Store setup metrics
            self.results["models"]["cline_optimal"] = {
                "setup_time": setup_time,
                "available": True,
                "test_response_length": len(test_response.get('message', {}).get('content', ''))
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up cline-optimal model: {str(e)}")
            return False
    
    def test_model_phi2(self, test_suite: List[Dict], suite_name: str) -> ModelMetrics:
        """Test the Phi-2 model with a test suite"""
        logger.info(f"Testing Phi-2 model with {suite_name} suite ({len(test_suite)} tests)")
        
        metrics = ModelMetrics(model_name="phi2_finetuned")
        response_times = []
        response_lengths = []
        token_speeds = []
        successful_tests = 0
        quality_scores = []
        
        for i, test_case in enumerate(test_suite):
            logger.info(f"  Test {i+1}/{len(test_suite)}: {test_case['prompt'][:50]}...")
            
            try:
                # Record resource usage before
                if GPU_MONITORING and len(GPUtil.getGPUs()) > 0:
                    gpu_before = GPUtil.getGPUs()[0].memoryUsed
                
                # Generate response
                start_time = time.time()
                result = self.phi2_pipeline(
                    test_case['prompt'],
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                    return_full_text=False
                )
                response_time = time.time() - start_time
                
                response = result[0]['generated_text']
                
                # Calculate metrics
                input_tokens = len(self.phi2_tokenizer.encode(test_case['prompt']))
                output_tokens = len(self.phi2_tokenizer.encode(response))
                tokens_per_second = output_tokens / response_time if response_time > 0 else 0
                
                # Quality assessment
                quality_score = self._assess_response_quality(test_case, response)
                
                # Success criteria
                is_successful = (
                    len(response) >= test_case.get('min_length', 10) and
                    response_time <= self.config.max_response_time
                )
                
                if is_successful:
                    successful_tests += 1
                    response_times.append(response_time)
                    response_lengths.append(len(response))
                    token_speeds.append(tokens_per_second)
                    quality_scores.append(quality_score)
                
                # Record detailed results
                test_result = {
                    "prompt": test_case['prompt'],
                    "response": response,
                    "response_time": response_time,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "tokens_per_second": tokens_per_second,
                    "quality_score": quality_score,
                    "successful": is_successful,
                    "category": test_case.get('category', 'unknown')
                }
                
                if GPU_MONITORING and len(GPUtil.getGPUs()) > 0:
                    gpu_after = GPUtil.getGPUs()[0].memoryUsed
                    test_result["gpu_memory_used_mb"] = gpu_after - gpu_before
                
                # Store in results
                if "phi2_finetuned" not in self.results["models"]:
                    self.results["models"]["phi2_finetuned"] = {}
                if suite_name not in self.results["models"]["phi2_finetuned"]:
                    self.results["models"]["phi2_finetuned"][suite_name] = []
                
                self.results["models"]["phi2_finetuned"][suite_name].append(test_result)
                
                # Use ASCII-only for logging to avoid Unicode errors
                logger.info(f"OK: {response_time:.2f}s, {len(response)} chars, Quality: {quality_score:.2f}")
                
            except Exception as e:
                logger.error(f"    ERROR: {str(e)}")
                # Record failed test
                if "phi2_finetuned" not in self.results["models"]:
                    self.results["models"]["phi2_finetuned"] = {}
                if suite_name not in self.results["models"]["phi2_finetuned"]:
                    self.results["models"]["phi2_finetuned"][suite_name] = []
                
                self.results["models"]["phi2_finetuned"][suite_name].append({
                    "prompt": test_case['prompt'],
                    "error": str(e),
                    "successful": False
                })
        
        # Calculate aggregate metrics
        if response_times:
            metrics.avg_response_time = np.mean(response_times)
            metrics.min_response_time = np.min(response_times)
            metrics.max_response_time = np.max(response_times)
            metrics.avg_tokens_per_second = np.mean(token_speeds)
            metrics.avg_response_length = np.mean(response_lengths)
            metrics.overall_quality_score = np.mean(quality_scores)
        
        metrics.success_rate = (successful_tests / len(test_suite)) * 100
        
        return metrics
    
    def test_model_cline_optimal(self, test_suite: List[Dict], suite_name: str) -> ModelMetrics:
        """Test the cline-optimal model with a test suite"""
        logger.info(f"Testing cline-optimal model with {suite_name} suite ({len(test_suite)} tests)")
        
        metrics = ModelMetrics(model_name="cline_optimal")
        response_times = []
        response_lengths = []
        token_speeds = []
        successful_tests = 0
        quality_scores = []
        
        for i, test_case in enumerate(test_suite):
            logger.info(f"  Test {i+1}/{len(test_suite)}: {test_case['prompt'][:50]}...")
            
            try:
                # Record resource usage before
                if GPU_MONITORING and len(GPUtil.getGPUs()) > 0:
                    gpu_before = GPUtil.getGPUs()[0].memoryUsed
                
                # Generate response
                start_time = time.time()
                result = ollama.chat(
                    model=self.config.cline_optimal_model,
                    messages=[{"role": "user", "content": test_case['prompt']}],
                    options={
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_new_tokens
                    }
                )
                response_time = time.time() - start_time
                
                response = result['message']['content']
                
                # Estimate token metrics (rough approximation for Ollama)
                input_tokens = len(test_case['prompt'].split())
                output_tokens = len(response.split())
                tokens_per_second = output_tokens / response_time if response_time > 0 else 0
                
                # Quality assessment
                quality_score = self._assess_response_quality(test_case, response)
                
                # Autonomy assessment if this is an autonomy test
                autonomy_score = 0.0
                if test_case.get('test_autonomy', False):
                    autonomy_score = self._assess_autonomy(response)
                
                # Success criteria
                is_successful = (
                    len(response) >= test_case.get('min_length', 10) and
                    response_time <= self.config.max_response_time
                )
                
                if is_successful:
                    successful_tests += 1
                    response_times.append(response_time)
                    response_lengths.append(len(response))
                    token_speeds.append(tokens_per_second)
                    quality_scores.append(quality_score)
                
                # Record detailed results
                test_result = {
                    "prompt": test_case['prompt'],
                    "response": response,
                    "response_time": response_time,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "tokens_per_second": tokens_per_second,
                    "quality_score": quality_score,
                    "autonomy_score": autonomy_score,
                    "successful": is_successful,
                    "category": test_case.get('category', 'unknown')
                }
                
                if GPU_MONITORING and len(GPUtil.getGPUs()) > 0:
                    gpu_after = GPUtil.getGPUs()[0].memoryUsed
                    test_result["gpu_memory_used_mb"] = gpu_after - gpu_before
                
                # Store in results
                if "cline_optimal" not in self.results["models"]:
                    self.results["models"]["cline_optimal"] = {}
                if suite_name not in self.results["models"]["cline_optimal"]:
                    self.results["models"]["cline_optimal"][suite_name] = []
                
                self.results["models"]["cline_optimal"][suite_name].append(test_result)
                
                # Use ASCII-only for logging to avoid Unicode errors
                logger.info(f"OK: {response_time:.2f}s, {len(response)} chars, Quality: {quality_score:.2f}")
                
            except Exception as e:
                logger.error(f"    ERROR: {str(e)}")
                # Record failed test
                if "cline_optimal" not in self.results["models"]:
                    self.results["models"]["cline_optimal"] = {}
                if suite_name not in self.results["models"]["cline_optimal"]:
                    self.results["models"]["cline_optimal"][suite_name] = []
                
                self.results["models"]["cline_optimal"][suite_name].append({
                    "prompt": test_case['prompt'],
                    "error": str(e),
                    "successful": False
                })
        
        # Calculate aggregate metrics
        if response_times:
            metrics.avg_response_time = np.mean(response_times)
            metrics.min_response_time = np.min(response_times)
            metrics.max_response_time = np.max(response_times)
            metrics.avg_tokens_per_second = np.mean(token_speeds)
            metrics.avg_response_length = np.mean(response_lengths)
            metrics.overall_quality_score = np.mean(quality_scores)
        
        metrics.success_rate = (successful_tests / len(test_suite)) * 100
        
        return metrics
    
    def _assess_response_quality(self, test_case: Dict, response: str) -> float:
        """Assess the quality of a response"""
        score = 0.0
        
        # Length check (basic quality indicator)
        min_length = test_case.get('min_length', 10)
        if len(response) >= min_length:
            score += 0.3
        
        # Keyword matching
        expected_keywords = test_case.get('expected_keywords', [])
        if expected_keywords:
            response_lower = response.lower()
            keyword_matches = sum(1 for keyword in expected_keywords if keyword.lower() in response_lower)
            keyword_score = keyword_matches / len(expected_keywords)
            score += keyword_score * 0.4
        
        # Task-specific quality assessment
        task_type = test_case.get('task_type', 'general')
        if task_type == 'coding':
            score += self._assess_code_quality(response) * 0.3
        elif task_type == 'reasoning':
            score += self._assess_reasoning_quality(response) * 0.3
        elif task_type == 'knowledge':
            score += self._assess_knowledge_quality(response) * 0.3
        else:
            score += 0.3  # Default score for general responses
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _assess_code_quality(self, response: str) -> float:
        """Assess the quality of code in the response"""
        score = 0.0
        response_lower = response.lower()
        
        # Check for code structure indicators
        code_indicators = ['def ', 'class ', 'import ', 'return ', 'if ', 'for ', 'while ']
        if any(indicator in response_lower for indicator in code_indicators):
            score += 0.5
        
        # Check for proper indentation (basic check)
        lines = response.split('\n')
        indented_lines = sum(1 for line in lines if line.startswith('    ') or line.startswith('\t'))
        if indented_lines > 0:
            score += 0.3
        
        # Check for comments
        if '#' in response or '"""' in response or "'''" in response:
            score += 0.2
        
        return min(score, 1.0)
    
    def _assess_reasoning_quality(self, response: str) -> float:
        """Assess the quality of reasoning in the response"""
        score = 0.0
        response_lower = response.lower()
        
        # Check for logical connectors
        logical_words = ['because', 'therefore', 'since', 'thus', 'hence', 'so', 'first', 'second', 'finally']
        if any(word in response_lower for word in logical_words):
            score += 0.4
        
        # Check for structured thinking
        if any(phrase in response_lower for phrase in ['step 1', 'step 2', 'first,', 'second,', 'next,']):
            score += 0.3
        
        # Check for numerical calculations
        if any(char.isdigit() for char in response):
            score += 0.3
        
        return min(score, 1.0)
    
    def _assess_knowledge_quality(self, response: str) -> float:
        """Assess the quality of knowledge in the response"""
        score = 0.0
        
        # Check response length (knowledge responses should be detailed)
        if len(response) > 100:
            score += 0.4
        elif len(response) > 50:
            score += 0.2
        
        # Check for technical terminology (presence of technical words)
        sentences = response.split('.')
        if len(sentences) > 2:  # Multi-sentence explanation
            score += 0.3
        
        # Check for examples
        if any(phrase in response.lower() for phrase in ['example', 'for instance', 'such as', 'e.g.']):
            score += 0.3
        
        return min(score, 1.0)
    
    def _assess_autonomy(self, response: str) -> float:
        """Assess how autonomous the response is (fewer questions = higher autonomy)"""
        question_count = response.count('?')
        
        # Check for phrases that indicate asking for clarification
        clarification_phrases = [
            'do you want', 'would you like', 'could you clarify', 'please specify',
            'what kind of', 'which type', 'more details', 'can you provide'
        ]
        
        clarification_count = sum(1 for phrase in clarification_phrases if phrase in response.lower())
        
        # Calculate autonomy score (inverse of questions/clarifications)
        total_issues = question_count + clarification_count
        if total_issues == 0:
            return 1.0
        elif total_issues <= 2:
            return 0.7
        elif total_issues <= 4:
            return 0.4
        else:
            return 0.1
    
    def run_speed_benchmark(self, model_name: str, iterations: int = 10) -> Dict:
        """Run a dedicated speed benchmark"""
        logger.info(f"Running speed benchmark for {model_name} ({iterations} iterations)")
        
        simple_prompt = "Hello world"
        times = []
        
        for i in range(iterations):
            try:
                if model_name == "phi2_finetuned" and hasattr(self, 'phi2_pipeline'):
                    start_time = time.time()
                    self.phi2_pipeline(simple_prompt, max_new_tokens=50, temperature=0.1)
                    response_time = time.time() - start_time
                elif model_name == "cline_optimal":
                    start_time = time.time()
                    ollama.chat(
                        model=self.config.cline_optimal_model,
                        messages=[{"role": "user", "content": simple_prompt}],
                        options={"temperature": 0.1, "num_predict": 50}
                    )
                    response_time = time.time() - start_time
                else:
                    continue
                    
                times.append(response_time)
                logger.info(f"  Iteration {i+1}: {response_time:.3f}s")
                
            except Exception as e:
                logger.error(f"  Iteration {i+1} failed: {str(e)}")
        
        if times:
            return {
                "avg_time": np.mean(times),
                "min_time": np.min(times),
                "max_time": np.max(times),
                "std_time": np.std(times),
                "successful_iterations": len(times),
                "total_iterations": iterations
            }
        else:
            return {"error": "No successful iterations"}
    
    def run_comprehensive_comparison(self, test_levels: List[str] = None) -> Dict:
        """Run the comprehensive comparison"""
        logger.info("Starting Comprehensive Model Comparison")
        print("=" * 80)
        
        if test_levels is None:
            test_levels = ["basic", "intermediate", "advanced", "autonomous"]
        
        # Setup models
        phi2_ready = self.setup_phi2_model()
        cline_ready = self.setup_cline_optimal()
        
        if not phi2_ready:
            logger.warning("Phi-2 model not available - running limited comparison")
        if not cline_ready:
            logger.error("Cline-optimal model not available - cannot proceed")
            return {"error": "Cline-optimal model not available"}
        
        # Run tests for each level
        comparison_results = {}
        
        for level in test_levels:
            if level not in self.test_suites:
                logger.warning(f"Test level '{level}' not found, skipping")
                continue
                
            logger.info(f"Running {level.upper()} tests")
            print("-" * 60)
            
            test_suite = self.test_suites[level]
            
            # Test both models if available
            if phi2_ready:
                phi2_metrics = self.test_model_phi2(test_suite, level)
                comparison_results[f"phi2_{level}"] = phi2_metrics.__dict__
            
            if cline_ready:
                cline_metrics = self.test_model_cline_optimal(test_suite, level)
                comparison_results[f"cline_{level}"] = cline_metrics.__dict__
        
        # Run speed benchmarks
        logger.info("Running speed benchmarks")
        print("-" * 60)
        
        if phi2_ready:
            phi2_speed = self.run_speed_benchmark("phi2_finetuned")
            comparison_results["phi2_speed"] = phi2_speed
        
        if cline_ready:
            cline_speed = self.run_speed_benchmark("cline_optimal")
            comparison_results["cline_speed"] = cline_speed
        
        # Generate summary
        summary = self._generate_comparison_summary(comparison_results)
        
        # Store results
        self.results["comparisons"] = comparison_results
        self.results["summary"] = summary
        
        # Save to file
        self.save_results()
        
        return self.results
    
    def _generate_comparison_summary(self, comparison_results: Dict) -> Dict:
        """Generate a comprehensive comparison summary"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "overall_winner": None,
            "speed_comparison": {},
            "quality_comparison": {},
            "strengths": {},
            "recommendations": []
        }
        
        # Speed comparison
        phi2_speed = comparison_results.get("phi2_speed", {})
        cline_speed = comparison_results.get("cline_speed", {})
        
        if phi2_speed.get("avg_time") and cline_speed.get("avg_time"):
            phi2_avg = phi2_speed["avg_time"]
            cline_avg = cline_speed["avg_time"]
            speed_diff = ((cline_avg - phi2_avg) / phi2_avg) * 100
            
            summary["speed_comparison"] = {
                "phi2_avg_time": phi2_avg,
                "cline_avg_time": cline_avg,
                "difference_percent": speed_diff,
                "faster_model": "phi2_finetuned" if phi2_avg < cline_avg else "cline_optimal"
            }
        
        # Quality comparison across test levels
        phi2_quality_scores = []
        cline_quality_scores = []
        
        for key, metrics in comparison_results.items():
            if key.startswith("phi2_") and not key.endswith("_speed"):
                if "overall_quality_score" in metrics:
                    phi2_quality_scores.append(metrics["overall_quality_score"])
            elif key.startswith("cline_") and not key.endswith("_speed"):
                if "overall_quality_score" in metrics:
                    cline_quality_scores.append(metrics["overall_quality_score"])
        
        if phi2_quality_scores and cline_quality_scores:
            phi2_avg_quality = np.mean(phi2_quality_scores)
            cline_avg_quality = np.mean(cline_quality_scores)
            
            summary["quality_comparison"] = {
                "phi2_avg_quality": phi2_avg_quality,
                "cline_avg_quality": cline_avg_quality,
                "difference": cline_avg_quality - phi2_avg_quality,
                "higher_quality": "phi2_finetuned" if phi2_avg_quality > cline_avg_quality else "cline_optimal"
            }
        
        # Overall winner determination
        speed_winner = summary.get("speed_comparison", {}).get("faster_model")
        quality_winner = summary.get("quality_comparison", {}).get("higher_quality")
        
        if speed_winner == quality_winner:
            summary["overall_winner"] = speed_winner
        else:
            summary["overall_winner"] = "tie"
        
        # Generate recommendations
        recommendations = []
        
        if summary.get("speed_comparison"):
            speed_diff = abs(summary["speed_comparison"]["difference_percent"])
            if speed_diff > 20:
                faster = summary["speed_comparison"]["faster_model"]
                recommendations.append(f"For speed-critical applications, use {faster} (>{speed_diff:.0f}% faster)")
        
        if summary.get("quality_comparison"):
            quality_diff = abs(summary["quality_comparison"]["difference"])
            if quality_diff > 0.1:
                better = summary["quality_comparison"]["higher_quality"]
                recommendations.append(f"For quality-critical applications, use {better} (higher response quality)")
        
        summary["recommendations"] = recommendations
        
        return summary
    
    def save_results(self):
        """Save comprehensive results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_comparison_{timestamp}.json"
        filepath = os.path.join(self.config.results_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {filepath}")
            
            # Also save a human-readable summary
            summary_file = filepath.replace('.json', '_summary.txt')
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(self._format_human_readable_summary())
            
            logger.info(f"Summary saved to {summary_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    def _format_human_readable_summary(self) -> str:
        """Format a human-readable summary"""
        output = []
        output.append("COMPREHENSIVE MODEL COMPARISON SUMMARY")
        output.append("=" * 50)
        output.append(f"Timestamp: {self.results['timestamp']}")
        output.append("")
        
        summary = self.results.get("summary", {})
        
        # Overall winner
        if summary.get("overall_winner"):
            output.append(f"🏆 Overall Winner: {summary['overall_winner']}")
            output.append("")
        
        # Speed comparison
        speed_comp = summary.get("speed_comparison", {})
        if speed_comp:
            output.append("⚡ SPEED COMPARISON")
            output.append("-" * 20)
            output.append(f"Phi-2 Fine-tuned: {speed_comp.get('phi2_avg_time', 'N/A'):.3f}s")
            output.append(f"Cline-Optimal: {speed_comp.get('cline_avg_time', 'N/A'):.3f}s")
            output.append(f"Difference: {speed_comp.get('difference_percent', 0):.1f}%")
            output.append(f"Faster Model: {speed_comp.get('faster_model', 'Unknown')}")
            output.append("")
        
        # Quality comparison  
        quality_comp = summary.get("quality_comparison", {})
        if quality_comp:
            output.append("💯 QUALITY COMPARISON")
            output.append("-" * 20)
            output.append(f"Phi-2 Fine-tuned: {quality_comp.get('phi2_avg_quality', 'N/A'):.3f}")
            output.append(f"Cline-Optimal: {quality_comp.get('cline_avg_quality', 'N/A'):.3f}")
            output.append(f"Difference: {quality_comp.get('difference', 0):.3f}")
            output.append(f"Higher Quality: {quality_comp.get('higher_quality', 'Unknown')}")
            output.append("")
        
        # Recommendations
        recommendations = summary.get("recommendations", [])
        if recommendations:
            output.append("📝 RECOMMENDATIONS")
            output.append("-" * 20)
            for i, rec in enumerate(recommendations, 1):
                output.append(f"{i}. {rec}")
            output.append("")
        
        # Detailed results by test level
        comparisons = self.results.get("comparisons", {})
        test_levels = ["basic", "intermediate", "advanced", "autonomous"]
        
        for level in test_levels:
            phi2_key = f"phi2_{level}"
            cline_key = f"cline_{level}"
            
            if phi2_key in comparisons or cline_key in comparisons:
                output.append(f"📊 {level.upper()} TEST RESULTS")
                output.append("-" * 30)
                
                if phi2_key in comparisons:
                    phi2_data = comparisons[phi2_key]
                    output.append(f"Phi-2 Fine-tuned:")
                    output.append(f"  Success Rate: {phi2_data.get('success_rate', 0):.1f}%")
                    output.append(f"  Avg Response Time: {phi2_data.get('avg_response_time', 0):.3f}s")
                    output.append(f"  Quality Score: {phi2_data.get('overall_quality_score', 0):.3f}")
                
                if cline_key in comparisons:
                    cline_data = comparisons[cline_key]
                    output.append(f"Cline-Optimal:")
                    output.append(f"  Success Rate: {cline_data.get('success_rate', 0):.1f}%")
                    output.append(f"  Avg Response Time: {cline_data.get('avg_response_time', 0):.3f}s")
                    output.append(f"  Quality Score: {cline_data.get('overall_quality_score', 0):.3f}")
                
                output.append("")
        
        return "\n".join(output)
    
    def print_live_summary(self):
        """Print a live summary to console"""
        print("\n" + "=" * 80)
        print("🎯 COMPREHENSIVE MODEL COMPARISON COMPLETE")
        print("=" * 80)
        
        summary = self.results.get("summary", {})
        
        # Overall winner
        if summary.get("overall_winner"):
            print(f"\n🏆 Overall Winner: {summary['overall_winner'].upper()}")
        
        # Speed comparison
        speed_comp = summary.get("speed_comparison", {})
        if speed_comp:
            print(f"\n⚡ Speed: {speed_comp.get('faster_model', 'Unknown').upper()} is faster")
            print(f"   Difference: {speed_comp.get('difference_percent', 0):.1f}%")
        
        # Quality comparison
        quality_comp = summary.get("quality_comparison", {})
        if quality_comp:
            print(f"\n💯 Quality: {quality_comp.get('higher_quality', 'Unknown').upper()} has higher quality")
            print(f"   Difference: {quality_comp.get('difference', 0):.3f}")
        
        # Recommendations
        recommendations = summary.get("recommendations", [])
        if recommendations:
            print(f"\n📝 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        print(f"\n📁 Detailed results saved to: {self.config.results_dir}")
        print("=" * 80)

def main():
    """Main function to run comprehensive comparison"""
    print("🚀 Comprehensive Model Comparison Tool")
    print("=" * 60)
    print("This will test your existing model against cline-optimal with multiple levels:")
    print("1. Basic tests (simple Q&A)")
    print("2. Intermediate tests (technical knowledge)")  
    print("3. Advanced tests (complex reasoning)")
    print("4. Autonomous tests (no questions asked)")
    print("5. Speed benchmarks")
    print()
    
    # Configuration
    config = TestConfig()
    
    # Allow user to customize test levels
    print("Available test levels: basic, intermediate, advanced, autonomous")
    user_levels = input("Enter test levels to run (comma-separated, or press Enter for all): ").strip()
    
    if user_levels:
        test_levels = [level.strip() for level in user_levels.split(",")]
    else:
        test_levels = ["basic", "intermediate", "advanced", "autonomous"]
    
    print(f"\nRunning tests: {', '.join(test_levels)}")
    print("=" * 60)
    
    # Run comparison
    comparison = ComprehensiveModelComparison(config)
    results = comparison.run_comprehensive_comparison(test_levels)
    
    # Print summary
    comparison.print_live_summary()
    
    return results

if __name__ == "__main__":
    main()
