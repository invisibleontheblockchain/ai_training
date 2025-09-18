#!/usr/bin/env python3
"""
Multi-Model Orchestrator
=======================
Intelligent routing system that selects the best model for each task based on
complexity, domain expertise, and performance characteristics.
"""

import asyncio
import json
import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import subprocess

# Import our research optimizer
from research_prompt_optimizer import ResearchPromptOptimizer, ResearchPromptConfig

class ModelType(Enum):
    """Available model types in the system"""
    CLINE_OPTIMAL = "cline-optimal"
    PHI2_BASE = "phi2-base"
    PHI2_FINETUNED = "phi2-finetuned"
    CODE_LLAMA = "code-llama"
    QWEN_CODER = "qwen-coder"
    OPENAI_GPT4 = "openai-gpt4"
    ANTHROPIC_CLAUDE = "anthropic-claude"

@dataclass
class ModelCapabilities:
    """Model capabilities and characteristics"""
    model_type: ModelType
    strengths: List[str]
    weaknesses: List[str]
    max_context: int
    avg_response_time: float
    quality_score: float
    specializations: List[str]
    cost_per_request: float
    local_available: bool

@dataclass
class TaskClassification:
    """Classification of a task for model selection"""
    complexity_level: int  # 1-5
    domain: str
    task_type: str
    estimated_tokens: int
    security_critical: bool
    performance_critical: bool
    creativity_required: bool
    code_generation: bool
    research_required: bool

class ModelOrchestrator:
    """Orchestrates multiple models for optimal task routing"""
    
    def __init__(self):
        self.setup_logging()
        self.research_optimizer = ResearchPromptOptimizer()
        self.load_model_configurations()
        self.performance_tracker = {}
        self.load_performance_history()
        
    def setup_logging(self):
        """Setup logging for the orchestrator"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('multi_model_orchestrator.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def load_model_configurations(self):
        """Load configurations for all available models"""
        self.models = {
            ModelType.CLINE_OPTIMAL: ModelCapabilities(
                model_type=ModelType.CLINE_OPTIMAL,
                strengths=["code_generation", "defi_protocols", "solidity", "security"],
                weaknesses=["creative_writing", "general_knowledge"],
                max_context=4096,
                avg_response_time=3.5,
                quality_score=0.85,
                specializations=["blockchain", "smart_contracts", "web3"],
                cost_per_request=0.0,  # Local model
                local_available=True
            ),
            
            ModelType.PHI2_FINETUNED: ModelCapabilities(
                model_type=ModelType.PHI2_FINETUNED,
                strengths=["fast_inference", "code_completion", "lightweight"],
                weaknesses=["complex_reasoning", "long_context"],
                max_context=2048,
                avg_response_time=1.2,
                quality_score=0.75,
                specializations=["python", "javascript", "api_development"],
                cost_per_request=0.0,  # Local model
                local_available=True
            ),
            
            ModelType.CODE_LLAMA: ModelCapabilities(
                model_type=ModelType.CODE_LLAMA,
                strengths=["code_generation", "debugging", "optimization"],
                weaknesses=["domain_specific_knowledge", "latest_frameworks"],
                max_context=16384,
                avg_response_time=8.0,
                quality_score=0.90,
                specializations=["python", "cpp", "java", "javascript"],
                cost_per_request=0.0,  # Local if available
                local_available=False  # Depends on setup
            ),
            
            ModelType.QWEN_CODER: ModelCapabilities(
                model_type=ModelType.QWEN_CODER,
                strengths=["multilingual_code", "documentation", "refactoring"],
                weaknesses=["specialized_domains", "performance_optimization"],
                max_context=8192,
                avg_response_time=5.5,
                quality_score=0.82,
                specializations=["python", "javascript", "typescript", "rust"],
                cost_per_request=0.0,  # Local if available
                local_available=False  # Depends on setup
            ),
            
            ModelType.OPENAI_GPT4: ModelCapabilities(
                model_type=ModelType.OPENAI_GPT4,
                strengths=["reasoning", "creativity", "general_knowledge", "complex_tasks"],
                weaknesses=["cost", "latency", "privacy"],
                max_context=128000,
                avg_response_time=15.0,
                quality_score=0.95,
                specializations=["general_programming", "architecture", "problem_solving"],
                cost_per_request=0.03,  # Approximate cost
                local_available=False
            ),
            
            ModelType.ANTHROPIC_CLAUDE: ModelCapabilities(
                model_type=ModelType.ANTHROPIC_CLAUDE,
                strengths=["safety", "reasoning", "long_context", "analysis"],
                weaknesses=["cost", "availability", "speed"],
                max_context=200000,
                avg_response_time=12.0,
                quality_score=0.93,
                specializations=["analysis", "architecture", "security_review"],
                cost_per_request=0.025,  # Approximate cost
                local_available=False
            )
        }
    
    def load_performance_history(self):
        """Load historical performance data"""
        try:
            if os.path.exists("model_performance_history.json"):
                with open("model_performance_history.json", 'r') as f:
                    self.performance_tracker = json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load performance history: {e}")
            self.performance_tracker = {}
    
    def save_performance_history(self):
        """Save performance data for future optimization"""
        try:
            with open("model_performance_history.json", 'w') as f:
                json.dump(self.performance_tracker, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Could not save performance history: {e}")
    
    def classify_task(self, task_description: str, context: Dict[str, Any] = None) -> TaskClassification:
        """Classify a task to determine optimal model selection"""
        
        # Analyze task complexity
        complexity_indicators = {
            "simple": ["hello", "basic", "simple", "quick", "small"],
            "medium": ["api", "function", "class", "module", "integrate"],
            "complex": ["architecture", "system", "optimize", "secure", "scalable"],
            "advanced": ["distributed", "microservices", "enterprise", "performance"],
            "expert": ["consensus", "cryptographic", "zero-knowledge", "formal_verification"]
        }
        
        task_lower = task_description.lower()
        complexity_level = 1
        
        for level, indicators in enumerate(complexity_indicators.values(), 1):
            if any(indicator in task_lower for indicator in indicators):
                complexity_level = level
        
        # Determine domain
        domain_keywords = {
            "blockchain": ["smart contract", "solidity", "defi", "nft", "token", "blockchain"],
            "web_development": ["api", "rest", "http", "web", "frontend", "backend"],
            "ai_ml": ["machine learning", "neural", "model", "training", "ai"],
            "devops": ["docker", "kubernetes", "deploy", "ci/cd", "infrastructure"],
            "data_science": ["data", "analysis", "pandas", "numpy", "visualization"],
            "security": ["secure", "authentication", "encryption", "vulnerability"]
        }
        
        domain = "general_programming"
        for dom, keywords in domain_keywords.items():
            if any(keyword in task_lower for keyword in keywords):
                domain = dom
                break
        
        # Determine task type
        task_type = "code_generation"
        if any(word in task_lower for word in ["review", "analyze", "explain"]):
            task_type = "analysis"
        elif any(word in task_lower for word in ["design", "architect", "plan"]):
            task_type = "architecture"
        elif any(word in task_lower for word in ["debug", "fix", "error"]):
            task_type = "debugging"
        elif any(word in task_lower for word in ["optimize", "improve", "performance"]):
            task_type = "optimization"
        
        # Estimate token requirements
        estimated_tokens = len(task_description.split()) * 50  # Rough estimate
        if complexity_level >= 4:
            estimated_tokens *= 3  # Complex tasks need more detailed responses
        
        # Determine special requirements
        security_critical = any(word in task_lower for word in ["secure", "auth", "encrypt", "vulnerability"])
        performance_critical = any(word in task_lower for word in ["performance", "optimize", "fast", "efficient"])
        creativity_required = any(word in task_lower for word in ["creative", "innovative", "novel", "unique"])
        code_generation = task_type in ["code_generation", "debugging", "optimization"]
        research_required = complexity_level >= 3 or domain in ["blockchain", "ai_ml"]
        
        return TaskClassification(
            complexity_level=complexity_level,
            domain=domain,
            task_type=task_type,
            estimated_tokens=estimated_tokens,
            security_critical=security_critical,
            performance_critical=performance_critical,
            creativity_required=creativity_required,
            code_generation=code_generation,
            research_required=research_required
        )
    
    def select_optimal_model(self, classification: TaskClassification, preferences: Dict[str, Any] = None) -> Tuple[ModelType, float]:
        """Select the optimal model based on task classification"""
        
        preferences = preferences or {}
        local_only = preferences.get("local_only", True)
        max_cost = preferences.get("max_cost", 0.01)
        max_response_time = preferences.get("max_response_time", 30.0)
        
        # Calculate scores for each available model
        model_scores = {}
        
        for model_type, capabilities in self.models.items():
            # Skip if not meeting basic requirements
            if local_only and not capabilities.local_available:
                continue
            if capabilities.cost_per_request > max_cost:
                continue
            if capabilities.avg_response_time > max_response_time:
                continue
            if classification.estimated_tokens > capabilities.max_context:
                continue
            
            # Check if model is actually available
            if not self.is_model_available(model_type):
                continue
            
            score = 0.0
            
            # Base quality score
            score += capabilities.quality_score * 0.3
            
            # Domain specialization bonus
            if classification.domain in capabilities.specializations:
                score += 0.2
            
            # Task type alignment
            task_strength_map = {
                "code_generation": ["code_generation", "debugging"],
                "analysis": ["reasoning", "analysis"],
                "architecture": ["reasoning", "complex_tasks"],
                "debugging": ["debugging", "code_generation"],
                "optimization": ["optimization", "performance_optimization"]
            }
            
            if classification.task_type in task_strength_map:
                required_strengths = task_strength_map[classification.task_type]
                strength_match = len(set(required_strengths) & set(capabilities.strengths))
                score += strength_match * 0.15
            
            # Complexity level alignment
            if classification.complexity_level <= 2 and capabilities.model_type == ModelType.PHI2_FINETUNED:
                score += 0.1  # Prefer fast model for simple tasks
            elif classification.complexity_level >= 4 and capabilities.model_type in [ModelType.OPENAI_GPT4, ModelType.ANTHROPIC_CLAUDE]:
                score += 0.15  # Prefer advanced models for complex tasks
            
            # Performance considerations
            if classification.performance_critical:
                # Prefer faster models
                time_score = max(0, (30 - capabilities.avg_response_time) / 30) * 0.1
                score += time_score
            
            # Cost efficiency for local models
            if capabilities.local_available:
                score += 0.1
            
            # Historical performance bonus
            model_history = self.performance_tracker.get(model_type.value, {})
            if model_history:
                avg_success_rate = model_history.get("success_rate", 0.5)
                score += avg_success_rate * 0.1
            
            model_scores[model_type] = score
        
        if not model_scores:
            # Fallback to cline-optimal if available
            if self.is_model_available(ModelType.CLINE_OPTIMAL):
                return ModelType.CLINE_OPTIMAL, 0.5
            else:
                raise ValueError("No suitable models available")
        
        # Select model with highest score
        best_model = max(model_scores.items(), key=lambda x: x[1])
        return best_model[0], best_model[1]
    
    def is_model_available(self, model_type: ModelType) -> bool:
        """Check if a model is actually available"""
        
        if model_type == ModelType.CLINE_OPTIMAL:
            try:
                import ollama
                models = ollama.list()
                return any("cline-optimal" in model['name'] for model in models.get('models', []))
            except:
                return False
        
        elif model_type == ModelType.PHI2_FINETUNED:
            # Check if fine-tuned Phi-2 model exists
            model_path = "./models/fine-tuned/phi2-optimized"
            return os.path.exists(model_path)
        
        elif model_type in [ModelType.CODE_LLAMA, ModelType.QWEN_CODER]:
            try:
                import ollama
                models = ollama.list()
                model_names = [model['name'] for model in models.get('models', [])]
                return any(model_type.value in name for name in model_names)
            except:
                return False
        
        elif model_type == ModelType.OPENAI_GPT4:
            return os.getenv("OPENAI_API_KEY") is not None
        
        elif model_type == ModelType.ANTHROPIC_CLAUDE:
            return os.getenv("ANTHROPIC_API_KEY") is not None
        
        return False
    
    async def execute_with_model(self, model_type: ModelType, prompt: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a prompt with the specified model"""
        
        start_time = time.time()
        options = options or {}
        
        try:
            if model_type == ModelType.CLINE_OPTIMAL:
                result = await self.execute_ollama_model("cline-optimal:latest", prompt, options)
            
            elif model_type == ModelType.PHI2_FINETUNED:
                result = await self.execute_phi2_model(prompt, options)
            
            elif model_type in [ModelType.CODE_LLAMA, ModelType.QWEN_CODER]:
                model_name = f"{model_type.value}:latest"
                result = await self.execute_ollama_model(model_name, prompt, options)
            
            elif model_type == ModelType.OPENAI_GPT4:
                result = await self.execute_openai_model(prompt, options)
            
            elif model_type == ModelType.ANTHROPIC_CLAUDE:
                result = await self.execute_anthropic_model(prompt, options)
            
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            execution_time = time.time() - start_time
            
            # Update performance tracking
            self.update_performance_metrics(model_type, execution_time, True, len(result.get('response', '')))
            
            result['execution_time'] = execution_time
            result['model_type'] = model_type.value
            result['success'] = True
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Update performance tracking for failure
            self.update_performance_metrics(model_type, execution_time, False, 0)
            
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'model_type': model_type.value
            }
    
    async def execute_ollama_model(self, model_name: str, prompt: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with Ollama model"""
        import ollama
        
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            options={
                'temperature': options.get('temperature', 0.3),
                'top_p': options.get('top_p', 0.9),
                'max_tokens': options.get('max_tokens', 4000)
            }
        )
        
        return {
            'response': response['message']['content'],
            'model': model_name
        }
    
    async def execute_phi2_model(self, prompt: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with fine-tuned Phi-2 model"""
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            
            model_path = "./models/fine-tuned/phi2-optimized"
            
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
            
            # Generate response
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
            
            result = pipe(
                prompt,
                max_length=options.get('max_tokens', 1000),
                temperature=options.get('temperature', 0.3),
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            return {
                'response': result[0]['generated_text'].replace(prompt, '').strip(),
                'model': 'phi2-finetuned'
            }
            
        except Exception as e:
            # Fallback to base Phi-2 if fine-tuned not available
            return await self.execute_ollama_model("phi", prompt, options)
    
    async def execute_openai_model(self, prompt: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with OpenAI GPT-4"""
        import openai
        
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=options.get('temperature', 0.3),
            max_tokens=options.get('max_tokens', 4000)
        )
        
        return {
            'response': response.choices[0].message.content,
            'model': 'gpt-4'
        }
    
    async def execute_anthropic_model(self, prompt: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with Anthropic Claude"""
        import anthropic
        
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        response = await client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=options.get('max_tokens', 4000),
            temperature=options.get('temperature', 0.3),
            messages=[{"role": "user", "content": prompt}]
        )
        
        return {
            'response': response.content[0].text,
            'model': 'claude-3-sonnet'
        }
    
    def update_performance_metrics(self, model_type: ModelType, execution_time: float, success: bool, response_length: int):
        """Update performance metrics for a model"""
        
        model_key = model_type.value
        
        if model_key not in self.performance_tracker:
            self.performance_tracker[model_key] = {
                "total_requests": 0,
                "successful_requests": 0,
                "total_time": 0,
                "avg_response_length": 0,
                "success_rate": 0,
                "avg_execution_time": 0
            }
        
        metrics = self.performance_tracker[model_key]
        metrics["total_requests"] += 1
        
        if success:
            metrics["successful_requests"] += 1
            metrics["total_time"] += execution_time
            
            # Update average response length
            prev_avg_length = metrics["avg_response_length"]
            metrics["avg_response_length"] = (
                (prev_avg_length * (metrics["successful_requests"] - 1) + response_length) 
                / metrics["successful_requests"]
            )
        
        # Update rates
        metrics["success_rate"] = metrics["successful_requests"] / metrics["total_requests"]
        metrics["avg_execution_time"] = (
            metrics["total_time"] / metrics["successful_requests"] 
            if metrics["successful_requests"] > 0 else 0
        )
        
        # Save updated metrics
        self.save_performance_history()
    
    async def intelligent_execute(self, task_description: str, preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a task with intelligent model selection and optimization"""
        
        # Classify the task
        classification = self.classify_task(task_description)
        
        self.logger.info(f"Task classified: {classification.task_type} in {classification.domain}, complexity {classification.complexity_level}")
        
        # Select optimal model
        selected_model, confidence_score = self.select_optimal_model(classification, preferences)
        
        self.logger.info(f"Selected model: {selected_model.value} (confidence: {confidence_score:.3f})")
        
        # Create optimized prompt based on task and model
        research_config = ResearchPromptConfig(
            task_type=classification.task_type,
            domain=classification.domain,
            complexity_level=classification.complexity_level,
            research_depth="medium" if classification.research_required else "shallow",
            code_quality_level="production",
            security_focus=classification.security_critical,
            performance_focus=classification.performance_critical,
            maintainability_focus=True,
            testing_focus=classification.code_generation
        )
        
        # Generate optimized prompt
        session = self.research_optimizer.create_research_session(task_description, research_config)
        optimized_prompt = session['generated_prompt']
        
        # Execute with selected model
        result = await self.execute_with_model(selected_model, optimized_prompt)
        
        # Add classification and selection info to result
        result.update({
            'task_classification': asdict(classification),
            'selected_model': selected_model.value,
            'model_confidence': confidence_score,
            'session_id': session['session_id']
        })
        
        return result
    
    def get_model_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report"""
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "models": {},
            "summary": {}
        }
        
        total_requests = 0
        total_successful = 0
        
        for model_key, metrics in self.performance_tracker.items():
            report["models"][model_key] = metrics.copy()
            total_requests += metrics["total_requests"]
            total_successful += metrics["successful_requests"]
        
        report["summary"] = {
            "total_requests": total_requests,
            "total_successful": total_successful,
            "overall_success_rate": total_successful / total_requests if total_requests > 0 else 0,
            "available_models": [model.value for model in self.models.keys() if self.is_model_available(model)]
        }
        
        return report

def main():
    """Main function for testing the multi-model orchestrator"""
    
    print("🤖 Multi-Model Orchestrator for Local Claude/Codex")
    print("=" * 60)
    
    async def test_orchestrator():
        orchestrator = ModelOrchestrator()
        
        # Test tasks of varying complexity
        test_tasks = [
            "Create a simple Hello World function in Python",
            "Build a secure REST API with JWT authentication and rate limiting",
            "Design a distributed microservices architecture for a high-traffic e-commerce platform",
            "Implement a smart contract for a DeFi lending protocol with flash loan protection"
        ]
        
        for i, task in enumerate(test_tasks, 1):
            print(f"\n--- Test {i}: {task[:50]}... ---")
            
            result = await orchestrator.intelligent_execute(task)
            
            if result['success']:
                print(f"✅ Model: {result['selected_model']}")
                print(f"⏱️ Time: {result['execution_time']:.1f}s")
                print(f"🎯 Confidence: {result['model_confidence']:.3f}")
            else:
                print(f"❌ Failed: {result['error']}")
        
        # Generate performance report
        report = orchestrator.get_model_performance_report()
        print(f"\n📊 Performance Report:")
        print(f"Available models: {report['summary']['available_models']}")
        print(f"Success rate: {report['summary']['overall_success_rate']*100:.1f}%")
    
    asyncio.run(test_orchestrator())

if __name__ == "__main__":
    main()
