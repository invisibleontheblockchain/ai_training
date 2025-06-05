#!/usr/bin/env python3
"""
Model Integration Script for Comprehensive Testing
=================================================

This script integrates the comprehensive testing framework with your specific models
(Phi-2 fine-tuned and cline-optimal) and provides easy-to-use testing commands.

Usage:
    python test_your_model.py --model phi2|cline-optimal|both [--level basic|intermediate|advanced|expert|autonomous|all]

Examples:
    python test_your_model.py --model phi2 --level basic
    python test_your_model.py --model both --level all
    python test_your_model.py --model cline-optimal --level autonomous

Created: June 4, 2025
"""

import sys
import os
import time
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Add the current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from comprehensive_model_test_prompt import ComprehensiveModelTester
except ImportError:
    print("Error: Could not import comprehensive_model_test_prompt.py")
    print("Make sure the file is in the same directory.")
    sys.exit(1)

# Model-specific imports
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from peft import PeftModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available - Phi-2 testing will be limited")

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️ Ollama not available - cline-optimal testing will be disabled")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelIntegration:
    """Integration class for testing different models"""
    
    def __init__(self):
        self.phi2_model = None
        self.phi2_tokenizer = None
        self.phi2_pipeline = None
        self.models_loaded = {}
        
    def load_phi2_model(self, model_path: str = "c:/AI_Training/models/fine-tuned/phi2-optimized") -> bool:
        """Load the fine-tuned Phi-2 model"""
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers not available - cannot load Phi-2 model")
            return False
            
        try:
            logger.info(f"Loading Phi-2 model from: {model_path}")
            
            # Check if path exists
            if not os.path.exists(model_path):
                logger.error(f"Model path does not exist: {model_path}")
                logger.info("Available model directories:")
                base_dir = Path("c:/AI_Training/models")
                if base_dir.exists():
                    for item in base_dir.rglob("*"):
                        if item.is_dir() and "phi" in item.name.lower():
                            logger.info(f"  - {item}")
                return False
            
            # Load tokenizer
            base_model_name = "microsoft/phi-2"
            self.phi2_tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Add pad token if missing
            if self.phi2_tokenizer.pad_token is None:
                self.phi2_tokenizer.pad_token = self.phi2_tokenizer.eos_token
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto"
            )
            
            # Load fine-tuned adapter
            self.phi2_model = PeftModel.from_pretrained(base_model, model_path)
            
            # Create pipeline for easier text generation
            self.phi2_pipeline = pipeline(
                "text-generation",
                model=self.phi2_model,
                tokenizer=self.phi2_tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                return_full_text=False
            )
            
            logger.info("✓ Phi-2 model loaded successfully")
            self.models_loaded['phi2'] = True
            return True
            
        except Exception as e:
            logger.error(f"Error loading Phi-2 model: {str(e)}")
            return False
    
    def test_cline_optimal_availability(self) -> bool:
        """Test if cline-optimal model is available"""
        if not OLLAMA_AVAILABLE:
            logger.error("Ollama not available - cannot test cline-optimal")
            return False
            
        try:
            logger.info("Testing cline-optimal model availability...")
            
            # List available models
            result = ollama.list()
            models = [model['name'] for model in result.get('models', [])]
            
            if 'cline-optimal:latest' not in models:
                logger.error("cline-optimal:latest not found in Ollama")
                logger.info("Available models:")
                for model in models:
                    logger.info(f"  - {model}")
                return False
            
            # Test connectivity with a simple prompt
            test_response = ollama.chat(
                model='cline-optimal:latest',
                messages=[{"role": "user", "content": "Hello, this is a test."}]
            )
            
            if test_response and 'message' in test_response:
                logger.info("✓ cline-optimal model is available and responding")
                self.models_loaded['cline-optimal'] = True
                return True
            else:
                logger.error("cline-optimal model not responding correctly")
                return False
                
        except Exception as e:
            logger.error(f"Error testing cline-optimal: {str(e)}")
            return False
    
    def create_phi2_function(self):
        """Create a function for testing Phi-2 model"""
        if not self.models_loaded.get('phi2', False):
            raise RuntimeError("Phi-2 model not loaded")
            
        def phi2_test_function(prompt: str) -> str:
            try:
                # Use pipeline for generation
                result = self.phi2_pipeline(prompt)
                response = result[0]['generated_text']
                
                # Clean up response (remove prompt if included)
                if response.startswith(prompt):
                    response = response[len(prompt):].strip()
                
                return response
                
            except Exception as e:
                logger.error(f"Error generating response with Phi-2: {str(e)}")
                return f"Error: {str(e)}"
        
        return phi2_test_function
    
    def create_cline_optimal_function(self):
        """Create a function for testing cline-optimal model"""
        if not self.models_loaded.get('cline-optimal', False):
            raise RuntimeError("cline-optimal model not available")
            
        def cline_optimal_test_function(prompt: str) -> str:
            try:
                result = ollama.chat(
                    model='cline-optimal:latest',
                    messages=[{"role": "user", "content": prompt}],
                    options={
                        "temperature": 0.7,
                        "num_predict": 512
                    }
                )
                
                return result['message']['content']
                
            except Exception as e:
                logger.error(f"Error generating response with cline-optimal: {str(e)}")
                return f"Error: {str(e)}"
        
        return cline_optimal_test_function

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test your AI models comprehensively")
    parser.add_argument("--model", choices=["phi2", "cline-optimal", "both"], 
                       required=True, help="Which model(s) to test")
    parser.add_argument("--level", choices=["basic", "intermediate", "advanced", "expert", "autonomous", "all"],
                       default="basic", help="Test difficulty level")
    parser.add_argument("--phi2-path", help="Custom path to Phi-2 model")
    parser.add_argument("--save-responses", action="store_true", 
                       help="Save detailed responses to files")
    
    args = parser.parse_args()
    
    print("🚀 Comprehensive Model Testing Framework")
    print("=" * 60)
    print(f"Testing model(s): {args.model}")
    print(f"Test level: {args.level}")
    print("=" * 60)
    
    # Initialize integration
    integration = ModelIntegration()
    tester = ComprehensiveModelTester()
    
    # Prepare models based on selection
    models_to_test = {}
    
    if args.model in ["phi2", "both"]:
        phi2_path = args.phi2_path or "c:/AI_Training/models/fine-tuned/phi2-optimized"
        if integration.load_phi2_model(phi2_path):
            models_to_test["phi2"] = integration.create_phi2_function()
        else:
            print("❌ Failed to load Phi-2 model")
            if args.model == "phi2":
                return 1
    
    if args.model in ["cline-optimal", "both"]:
        if integration.test_cline_optimal_availability():
            models_to_test["cline-optimal"] = integration.create_cline_optimal_function()
        else:
            print("❌ Failed to access cline-optimal model")
            if args.model == "cline-optimal":
                return 1
    
    if not models_to_test:
        print("❌ No models available for testing")
        return 1
    
    # Determine test levels
    if args.level == "all":
        test_levels = ["basic", "intermediate", "advanced", "autonomous"]
    else:
        test_levels = [args.level]
    
    # Run tests for each model
    all_results = {}
    
    for model_name, model_function in models_to_test.items():
        print(f"\n🧪 Testing {model_name}...")
        print("-" * 40)
        
        try:
            results = tester.run_all_tests(model_function, test_levels)
            all_results[model_name] = results
            
            # Generate report for this model
            report = tester.generate_report(results)
            print(f"\n📊 RESULTS FOR {model_name.upper()}")
            print(report)
            
            # Save results
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            results_file = f"test_results_{model_name}_{timestamp}.json"
            tester.save_results(results, results_file)
            
            print(f"💾 Results saved to: {results_file}")
            
        except Exception as e:
            logger.error(f"Error testing {model_name}: {str(e)}")
            print(f"❌ Error testing {model_name}: {str(e)}")
    
    # If testing both models, create comparison
    if len(all_results) > 1:
        print("\n📈 MODEL COMPARISON")
        print("=" * 60)
        
        comparison_data = {}
        
        for model_name, model_results in all_results.items():
            for suite_name, suite_results in model_results.items():
                if suite_name not in comparison_data:
                    comparison_data[suite_name] = {}
                
                comparison_data[suite_name][model_name] = {
                    "pass_rate": suite_results.passed_tests / suite_results.total_tests,
                    "avg_quality": suite_results.average_quality,
                    "avg_autonomy": suite_results.average_autonomy,
                    "avg_time": suite_results.average_response_time
                }
        
        # Print comparison table
        for suite_name, suite_data in comparison_data.items():
            print(f"\n{suite_name.upper()} SUITE COMPARISON:")
            print("-" * 40)
            
            for model_name, metrics in suite_data.items():
                print(f"{model_name:15} | "
                      f"Pass: {metrics['pass_rate']:.1%} | "
                      f"Quality: {metrics['avg_quality']:.2f} | "
                      f"Autonomy: {metrics['avg_autonomy']:.2f} | "
                      f"Time: {metrics['avg_time']:.1f}s")
        
        # Save comparison
        comparison_file = f"model_comparison_{time.strftime('%Y%m%d_%H%M%S')}.json"
        comparison_path = Path("test_results") / comparison_file
        comparison_path.parent.mkdir(exist_ok=True)
        
        import json
        with open(comparison_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        print(f"\n💾 Comparison saved to: {comparison_path}")
    
    print("\n✅ Testing completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
