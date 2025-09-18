#!/usr/bin/env python3
"""
DeFi Continuous Learning Tester
===============================
Real-time testing environment that monitors cline-optimal's performance
on DeFi development tasks and automatically generates training data
from failures to continuously improve the model.
"""

import ollama
import time
import json
import os
import subprocess
import tempfile
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import queue
import hashlib

@dataclass
class TestResult:
    """Comprehensive test result structure"""
    test_id: str
    prompt: str
    response: str
    timestamp: datetime
    
    # Performance Metrics
    response_time: float
    response_length: int
    
    # Validation Results
    compilation_success: bool = False
    compilation_errors: List[str] = None
    security_score: float = 0.0
    security_issues: List[str] = None
    gas_efficiency: float = 0.0
    integration_success: bool = False
    integration_errors: List[str] = None
    
    # Quality Scores
    code_quality_score: float = 0.0
    autonomy_score: float = 0.0
    overall_score: float = 0.0
    
    # Learning Data
    failure_categories: List[str] = None
    improvement_suggestions: List[str] = None
    training_data_generated: bool = False

@dataclass
class LearningMetrics:
    """Track learning progress over time"""
    session_id: str
    start_time: datetime
    total_tests: int = 0
    successful_tests: int = 0
    
    # Performance Trends
    avg_compilation_rate: float = 0.0
    avg_security_score: float = 0.0
    avg_gas_efficiency: float = 0.0
    avg_integration_rate: float = 0.0
    avg_overall_score: float = 0.0
    
    # Learning Progress
    improvement_rate: float = 0.0
    training_sessions_triggered: int = 0
    patterns_learned: List[str] = None

class DeFiTaskGenerator:
    """Generates progressive DeFi development tasks"""
    
    def __init__(self):
        self.difficulty_levels = {
            1: "Basic Contract Development",
            2: "Security-Enhanced Contracts", 
            3: "Multi-Contract Integration",
            4: "Complex DeFi Protocols",
            5: "Production-Ready Suite"
        }
        
        self.task_templates = self._initialize_task_templates()
    
    def _initialize_task_templates(self) -> Dict[int, List[Dict]]:
        """Initialize task templates for each difficulty level"""
        return {
            1: [  # Basic Contract Development
                {
                    "name": "Simple Oracle",
                    "prompt": "Create a basic price oracle contract that stores and retrieves asset prices with proper access control",
                    "expected_features": ["price storage", "access control", "price updates", "getter functions"],
                    "security_requirements": ["basic access control"],
                    "gas_target": 50000
                },
                {
                    "name": "Basic Token",
                    "prompt": "Implement an ERC20 token with mint and burn functionality",
                    "expected_features": ["ERC20 compliance", "mint function", "burn function", "total supply tracking"],
                    "security_requirements": ["overflow protection"],
                    "gas_target": 45000
                }
            ],
            2: [  # Security-Enhanced Contracts
                {
                    "name": "Secure Lending Pool",
                    "prompt": "Create a lending pool contract with reentrancy protection and proper collateral management",
                    "expected_features": ["lending", "borrowing", "collateral tracking", "interest calculation"],
                    "security_requirements": ["reentrancy protection", "overflow protection", "access control"],
                    "gas_target": 80000
                },
                {
                    "name": "Protected DEX",
                    "prompt": "Implement a simple DEX with MEV protection and slippage controls",
                    "expected_features": ["token swapping", "liquidity pools", "slippage protection"],
                    "security_requirements": ["MEV protection", "reentrancy protection", "price manipulation protection"],
                    "gas_target": 120000
                }
            ],
            3: [  # Multi-Contract Integration
                {
                    "name": "Oracle + DEX Integration",
                    "prompt": "Create an oracle contract and DEX that uses the oracle for price feeds with proper integration",
                    "expected_features": ["oracle contract", "DEX contract", "price feed integration", "fallback mechanisms"],
                    "security_requirements": ["oracle manipulation protection", "integration security"],
                    "gas_target": 150000
                },
                {
                    "name": "Lending + Liquidation System",
                    "prompt": "Build a lending protocol with automated liquidation using price oracles",
                    "expected_features": ["lending pool", "liquidation engine", "oracle integration", "health factor calculation"],
                    "security_requirements": ["liquidation protection", "oracle security", "flash loan protection"],
                    "gas_target": 200000
                }
            ],
            4: [  # Complex DeFi Protocols
                {
                    "name": "Yield Farming Vault",
                    "prompt": "Create a yield farming vault with strategy management and auto-compounding",
                    "expected_features": ["vault contract", "strategy interface", "auto-compounding", "fee management"],
                    "security_requirements": ["strategy security", "vault protection", "admin controls"],
                    "gas_target": 250000
                },
                {
                    "name": "AMM with Concentrated Liquidity",
                    "prompt": "Implement an AMM with concentrated liquidity similar to Uniswap V3",
                    "expected_features": ["concentrated liquidity", "tick management", "fee tiers", "position NFTs"],
                    "security_requirements": ["math overflow protection", "position security", "fee calculation accuracy"],
                    "gas_target": 300000
                }
            ],
            5: [  # Production-Ready Suite
                {
                    "name": "Complete DeFi Suite",
                    "prompt": """Create a complete DeFi suite with Oracle, DEX, Lending, Vaults, and Staking protocols.
                    
                    Requirements:
                    1. Oracle Protocol: Real-time price feeds with TWAP
                    2. DEX Protocol: AMM with liquidity pools and routing
                    3. Lending Protocol: Collateralized lending with liquidations
                    4. Vault Protocol: Yield farming with strategy management
                    5. Staking Protocol: Liquid staking with rewards
                    
                    All contracts must be production-ready with:
                    - Comprehensive security measures
                    - Gas optimization
                    - Proper integration between protocols
                    - Emergency controls
                    - Upgrade mechanisms
                    """,
                    "expected_features": [
                        "oracle system", "DEX functionality", "lending protocol", 
                        "vault management", "staking system", "cross-protocol integration"
                    ],
                    "security_requirements": [
                        "comprehensive security audit", "reentrancy protection", 
                        "oracle manipulation protection", "flash loan protection",
                        "admin controls", "emergency mechanisms"
                    ],
                    "gas_target": 500000
                }
            ]
        }
    
    def get_task(self, difficulty: int, task_index: int = 0) -> Dict:
        """Get a specific task by difficulty and index"""
        if difficulty not in self.task_templates:
            raise ValueError(f"Invalid difficulty level: {difficulty}")
        
        tasks = self.task_templates[difficulty]
        if task_index >= len(tasks):
            task_index = task_index % len(tasks)
        
        task = tasks[task_index].copy()
        task["difficulty"] = difficulty
        task["task_id"] = f"level_{difficulty}_task_{task_index}"
        return task
    
    def get_progressive_tasks(self, start_level: int = 1, end_level: int = 5) -> List[Dict]:
        """Get a series of progressive tasks"""
        tasks = []
        for level in range(start_level, end_level + 1):
            level_tasks = self.task_templates[level]
            for i, task in enumerate(level_tasks):
                task_copy = task.copy()
                task_copy["difficulty"] = level
                task_copy["task_id"] = f"level_{level}_task_{i}"
                tasks.append(task_copy)
        return tasks

class SolidityValidator:
    """Validates Solidity code for compilation, security, and gas efficiency"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.setup_validation_environment()
    
    def setup_validation_environment(self):
        """Setup validation tools and environment"""
        # Create basic hardhat project structure
        project_dir = Path(self.temp_dir) / "validation_project"
        project_dir.mkdir(exist_ok=True)
        
        # Create package.json
        package_json = {
            "name": "defi-validation",
            "version": "1.0.0",
            "devDependencies": {
                "hardhat": "^2.19.0",
                "@nomicfoundation/hardhat-toolbox": "^4.0.0",
                "@openzeppelin/contracts": "^5.3.0"
            }
        }
        
        with open(project_dir / "package.json", "w") as f:
            json.dump(package_json, f, indent=2)
        
        # Create hardhat config
        hardhat_config = '''
require("@nomicfoundation/hardhat-toolbox");

module.exports = {
  solidity: {
    version: "0.8.30",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200
      }
    }
  }
};
'''
        with open(project_dir / "hardhat.config.js", "w") as f:
            f.write(hardhat_config)
        
        # Create contracts directory
        (project_dir / "contracts").mkdir(exist_ok=True)
        
        self.project_dir = project_dir
    
    def validate_solidity_code(self, code: str, task_info: Dict) -> Tuple[bool, List[str], Dict]:
        """Validate Solidity code and return results"""
        try:
            # Extract contract name from code
            contract_name = self._extract_contract_name(code)
            if not contract_name:
                return False, ["Could not extract contract name"], {}
            
            # Write contract to file
            contract_file = self.project_dir / "contracts" / f"{contract_name}.sol"
            with open(contract_file, "w") as f:
                f.write(code)
            
            # Try compilation
            compilation_success, compilation_errors = self._compile_contract(contract_name)
            
            # Security analysis (basic)
            security_score, security_issues = self._analyze_security(code)
            
            # Gas estimation (basic)
            gas_efficiency = self._estimate_gas_efficiency(code, task_info.get("gas_target", 100000))
            
            # Feature completeness check
            feature_score = self._check_feature_completeness(code, task_info.get("expected_features", []))
            
            validation_results = {
                "compilation_success": compilation_success,
                "compilation_errors": compilation_errors,
                "security_score": security_score,
                "security_issues": security_issues,
                "gas_efficiency": gas_efficiency,
                "feature_completeness": feature_score
            }
            
            return compilation_success, compilation_errors, validation_results
            
        except Exception as e:
            return False, [f"Validation error: {str(e)}"], {}
    
    def _extract_contract_name(self, code: str) -> Optional[str]:
        """Extract contract name from Solidity code"""
        import re
        match = re.search(r'contract\s+(\w+)', code)
        return match.group(1) if match else None
    
    def _compile_contract(self, contract_name: str) -> Tuple[bool, List[str]]:
        """Attempt to compile the contract"""
        try:
            # Change to project directory
            original_dir = os.getcwd()
            os.chdir(self.project_dir)
            
            # Run hardhat compile
            result = subprocess.run(
                ["npx", "hardhat", "compile"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            os.chdir(original_dir)
            
            if result.returncode == 0:
                return True, []
            else:
                errors = result.stderr.split('\n') if result.stderr else ["Unknown compilation error"]
                return False, errors
                
        except subprocess.TimeoutExpired:
            return False, ["Compilation timeout"]
        except Exception as e:
            return False, [f"Compilation error: {str(e)}"]
    
    def _analyze_security(self, code: str) -> Tuple[float, List[str]]:
        """Basic security analysis of Solidity code"""
        security_score = 1.0
        issues = []
        
        # Check for common security patterns
        security_checks = [
            ("reentrancy protection", ["nonReentrant", "ReentrancyGuard"], 0.3),
            ("overflow protection", ["SafeMath", "unchecked"], 0.2),
            ("access control", ["onlyOwner", "AccessControl", "modifier"], 0.2),
            ("input validation", ["require(", "revert("], 0.15),
            ("event emission", ["emit "], 0.1),
            ("proper visibility", ["external", "internal", "private"], 0.05)
        ]
        
        for check_name, patterns, weight in security_checks:
            if not any(pattern in code for pattern in patterns):
                security_score -= weight
                issues.append(f"Missing {check_name}")
        
        return max(0.0, security_score), issues
    
    def _estimate_gas_efficiency(self, code: str, target_gas: int) -> float:
        """Estimate gas efficiency based on code patterns"""
        # Basic heuristic for gas efficiency
        gas_score = 1.0
        
        # Penalize inefficient patterns
        inefficient_patterns = [
            ("storage in loops", ["for", "storage"], 0.2),
            ("repeated external calls", ["call", "delegatecall"], 0.1),
            ("large arrays", ["uint256[]", "address[]"], 0.1),
            ("string operations", ["string", "bytes"], 0.05)
        ]
        
        for pattern_name, keywords, penalty in inefficient_patterns:
            if all(keyword in code for keyword in keywords):
                gas_score -= penalty
        
        return max(0.0, gas_score)
    
    def _check_feature_completeness(self, code: str, expected_features: List[str]) -> float:
        """Check if code implements expected features"""
        if not expected_features:
            return 1.0
        
        implemented_features = 0
        for feature in expected_features:
            # Simple keyword matching for feature detection
            feature_keywords = feature.lower().split()
            if any(keyword in code.lower() for keyword in feature_keywords):
                implemented_features += 1
        
        return implemented_features / len(expected_features)
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Warning: Could not clean up temp directory: {e}")

class ContinuousLearningTester:
    """Main testing framework with continuous learning capabilities"""
    
    def __init__(self):
        self.task_generator = DeFiTaskGenerator()
        self.validator = SolidityValidator()
        self.test_results: List[TestResult] = []
        self.learning_metrics = LearningMetrics(
            session_id=self._generate_session_id(),
            start_time=datetime.now()
        )
        
        # Learning configuration
        self.learning_config = {
            "failure_threshold": 0.7,  # Trigger learning if score < 70%
            "improvement_target": 0.05,  # Target 5% improvement per session
            "training_batch_size": 10,  # Number of failures before training
            "max_training_sessions": 5  # Max training sessions per run
        }
        
        # Training data queue
        self.training_queue = queue.Queue()
        self.training_thread = None
        self.training_active = False
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"defi_learning_{timestamp}"
    
    def run_progressive_test(self, start_level: int = 1, end_level: int = 5) -> Dict:
        """Run progressive testing from basic to advanced"""
        print(f"🚀 Starting Progressive DeFi Testing (Levels {start_level}-{end_level})")
        print("=" * 80)
        
        # Get progressive tasks
        tasks = self.task_generator.get_progressive_tasks(start_level, end_level)
        
        # Start training thread
        self._start_training_thread()
        
        try:
            for i, task in enumerate(tasks):
                print(f"\n📋 Task {i+1}/{len(tasks)}: {task['name']} (Level {task['difficulty']})")
                print(f"Prompt: {task['prompt'][:100]}...")
                
                # Run test
                result = self._run_single_test(task)
                self.test_results.append(result)
                
                # Update metrics
                self._update_learning_metrics(result)
                
                # Display results
                self._display_test_result(result)
                
                # Check if training is needed
                if result.overall_score < self.learning_config["failure_threshold"]:
                    self._queue_training_data(task, result)
                
                # Brief pause between tests
                time.sleep(1)
            
            # Final analysis
            final_metrics = self._generate_final_analysis()
            
            # Save results
            self._save_results()
            
            return final_metrics
            
        finally:
            self._stop_training_thread()
            self.validator.cleanup()
    
    def _run_single_test(self, task: Dict) -> TestResult:
        """Run a single test and return comprehensive results"""
        start_time = time.time()
        
        try:
            # Get response from cline-optimal
            response = ollama.chat(
                model='cline-optimal:latest',
                messages=[{"role": "user", "content": task['prompt']}],
                options={
                    "temperature": 0.7,
                    "num_predict": 2048  # Allow longer responses for complex tasks
                }
            )
            
            response_time = time.time() - start_time
            response_text = response['message']['content']
            
            # Validate the response
            compilation_success, compilation_errors, validation_results = self.validator.validate_solidity_code(
                response_text, task
            )
            
            # Calculate scores
            autonomy_score = self._calculate_autonomy_score(response_text)
            code_quality_score = self._calculate_code_quality_score(response_text, task)
            overall_score = self._calculate_overall_score(validation_results, autonomy_score, code_quality_score)
            
            # Analyze failures
            failure_categories = self._analyze_failure_categories(validation_results)
            improvement_suggestions = self._generate_improvement_suggestions(validation_results, task)
            
            # Create test result
            result = TestResult(
                test_id=f"{task['task_id']}_{int(time.time())}",
                prompt=task['prompt'],
                response=response_text,
                timestamp=datetime.now(),
                response_time=response_time,
                response_length=len(response_text),
                compilation_success=compilation_success,
                compilation_errors=compilation_errors or [],
                security_score=validation_results.get('security_score', 0.0),
                security_issues=validation_results.get('security_issues', []),
                gas_efficiency=validation_results.get('gas_efficiency', 0.0),
                integration_success=validation_results.get('feature_completeness', 0.0) > 0.8,
                integration_errors=[],
                code_quality_score=code_quality_score,
                autonomy_score=autonomy_score,
                overall_score=overall_score,
                failure_categories=failure_categories,
                improvement_suggestions=improvement_suggestions
            )
            
            return result
            
        except Exception as e:
            # Handle errors gracefully
            return TestResult(
                test_id=f"{task['task_id']}_error_{int(time.time())}",
                prompt=task['prompt'],
                response=f"Error: {str(e)}",
                timestamp=datetime.now(),
                response_time=time.time() - start_time,
                response_length=0,
                compilation_success=False,
                compilation_errors=[str(e)],
                overall_score=0.0,
                failure_categories=["system_error"],
                improvement_suggestions=["Fix system error"]
            )
    
    def _calculate_autonomy_score(self, response: str) -> float:
        """Calculate autonomy score based on response characteristics"""
        # Count questions and clarification requests
        question_count = response.count('?')
        clarification_phrases = [
            'do you want', 'would you like', 'could you clarify', 'please specify',
            'what kind of', 'which type', 'more details', 'can you provide'
        ]
        
        clarification_count = sum(1 for phrase in clarification_phrases if phrase in response.lower())
        total_issues = question_count + clarification_count
        
        if total_issues == 0:
            return 1.0
        elif total_issues <= 2:
            return 0.8
        elif total_issues <= 4:
            return 0.6
        else:
            return 0.3
    
    def _calculate_code_quality_score(self, response: str, task: Dict) -> float:
        """Calculate code quality score"""
        score = 0.0
        
        # Check for code structure
        if 'contract ' in response and '{' in response:
            score += 0.3
        
        # Check for proper Solidity syntax
        solidity_keywords = ['function', 'modifier', 'event', 'mapping', 'struct']
        if any(keyword in response for keyword in solidity_keywords):
            score += 0.2
        
        # Check for documentation
        if '/**' in response or '//' in response or '///' in response:
            score += 0.2
        
        # Check for error handling
        if 'require(' in response or 'revert(' in response:
            score += 0.15
        
        # Check for events
        if 'emit ' in response:
            score += 0.15
        
        return min(score, 1.0)
    
    def _calculate_overall_score(self, validation_results: Dict, autonomy_score: float, code_quality_score: float) -> float:
        """Calculate overall performance score"""
        weights = {
            'compilation': 0.25,
            'security': 0.25,
            'gas_efficiency': 0.15,
            'feature_completeness': 0.15,
            'autonomy': 0.1,
            'code_quality': 0.1
        }
        
        scores = {
            'compilation': 1.0 if validation_results.get('compilation_success', False) else 0.0,
            'security': validation_results.get('security_score', 0.0),
            'gas_efficiency': validation_results.get('gas_efficiency', 0.0),
            'feature_completeness': validation_results.get('feature_completeness', 0.0),
            'autonomy': autonomy_score,
            'code_quality': code_quality_score
        }
        
        overall_score = sum(scores[metric] * weights[metric] for metric in weights)
        return overall_score
    
    def _analyze_failure_categories(self, validation_results: Dict) -> List[str]:
        """Analyze and categorize failures"""
        categories = []
        
        if not validation_results.get('compilation_success', False):
            categories.append('compilation_failure')
        
        if validation_results.get('security_score', 1.0) < 0.7:
            categories.append('security_issues')
        
        if validation_results.get('gas_efficiency', 1.0) < 0.8:
            categories.append('gas_inefficiency')
        
        if validation_results.get('feature_completeness', 1.0) < 0.8:
            categories.append('incomplete_features')
        
        return categories
    
    def _generate_improvement_suggestions(self, validation_results: Dict, task: Dict) -> List[str]:
        """Generate specific improvement suggestions"""
        suggestions = []
        
        if not validation_results.get('compilation_success', False):
            suggestions.append("Fix compilation errors - check syntax and imports")
        
        if validation_results.get('security_score', 1.0) < 0.7:
            suggestions.extend([
                "Add reentrancy protection using ReentrancyGuard",
                "Implement proper access control with modifiers",
                "Add input validation with require statements"
            ])
        
        if validation_results.get('gas_efficiency', 1.0) < 0.8:
            suggestions.extend([
                "Optimize storage usage and avoid storage operations in loops",
                "Use memory instead of storage where possible",
                "Consider using assembly for gas-critical operations"
            ])
        
        if validation_results.get('feature_completeness', 1.0) < 0.8:
            missing_features = task.get('expected_features', [])
            suggestions.append(f"Implement missing features: {', '.join(missing_features)}")
        
        return suggestions
    
    def _update_learning_metrics(self, result: TestResult):
        """Update learning metrics with new test result"""
        self.learning_metrics.total_tests += 1
        
        if result.overall_score > 0.8:
            self.learning_metrics.successful_tests += 1
        
        # Update averages
        total_tests = self.learning_metrics.total_tests
        self.learning_metrics.avg_compilation_rate = (
            (self.learning_metrics.avg_compilation_rate * (total_tests - 1) + 
             (1.0 if result.compilation_success else 0.0)) / total_tests
        )
        
        self.learning_metrics.avg_security_score = (
            (self.learning_metrics.avg_security_score * (total_tests - 1) + 
             result.security_score) / total_tests
        )
        
        self.learning_metrics.avg_gas_efficiency = (
            (self.learning_metrics.avg_gas_efficiency * (total_tests - 1) + 
             result.gas_efficiency) / total_tests
        )
        
        self.learning_metrics.avg_overall_score = (
            (self.learning_metrics.avg_overall_score * (total_tests - 1) + 
             result.overall_score) / total_tests
        )
        
        # Calculate improvement rate
        if total_tests > 1:
            recent_scores = [r.overall_score for r in self.test_results[-5:]]  # Last 5 tests
            if len(recent_scores) >= 2:
                self.learning_metrics.improvement_rate = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
    
    def _display_test_result(self, result: TestResult):
        """Display test result in a formatted way"""
        status = "✅ PASS" if result.overall_score > 0.8 else "❌ FAIL"
        print(f"  {status} Overall Score: {result.overall_score:.2f}")
        print(f"    ├─ Compilation: {'✅' if result.compilation_success else '❌'}")
        print(f"    ├─ Security: {result.security_score:.2f}")
        print(f"    ├─ Gas Efficiency: {result.gas_efficiency:.2f}")
        print(f"    ├─ Autonomy: {result.autonomy_score:.2f}")
        print(f"    └─ Response Time: {result.response_time:.2f}s")
        
        if result.failure_categories:
            print(f"    Issues: {', '.join(result.failure_categories)}")
    
    def _queue_training_data(self, task: Dict, result: TestResult):
        """Queue training data for continuous learning"""
        training_data = {
            "task": task,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
        self.training_queue.put(training_data)
        print(f"    📚 Queued for training (Queue size: {self.training_queue.qsize()})")
    
    def _start_training_thread(self):
        """Start background training thread"""
        self.training_active = True
        self.training_thread = threading.Thread(target=self._training_worker)
        self.training_thread.daemon = True
        self.training_thread.start()
    
    def _stop_training_thread(self):
        """Stop background training thread"""
        self.training_active = False
        if self.training_thread:
            self.training_thread.join(timeout=5)
    
    def _training_worker(self):
        """Background worker for processing training data"""
        training_batch = []
        
        while self.training_active:
            try:
                # Get training data with timeout
                training_data = self.training_queue.get(timeout=1)
                training_batch.append(training_data)
                
                # Process batch when it reaches target size
                if len(training_batch) >= self.learning_config["training_batch_size"]:
                    self._process_training_batch(training_batch)
                    training_batch = []
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Training worker error: {e}")
    
    def _process_training_batch(self, training_batch: List[Dict]):
        """Process a batch of training data"""
        print(f"\n🧠 Processing training batch ({len(training_batch)} examples)")
        
        # Analyze failure patterns
        failure_patterns = self._analyze_failure_patterns(training_batch)
        
        # Generate training examples
        training_examples = self._generate_training_examples(training_batch)
        
        # Save training data
        self._save_training_data(training_examples, failure_patterns)
        
        self.learning_metrics.training_sessions_triggered += 1
        print(f"✅ Training data generated and saved")
    
    def _analyze_failure_patterns(self, training_batch: List[Dict]) -> Dict:
        """Analyze patterns in failures"""
        patterns = {
            "common_errors": {},
            "security_gaps": [],
            "feature_gaps": [],
            "performance_issues": []
        }
        
        for item in training_batch:
            result = item["result"]
            
            # Count error types
            for category in result.failure_categories:
                patterns["common_errors"][category] = patterns["common_errors"].get(category, 0) + 1
            
            # Collect security issues
            if result.security_issues:
                patterns["security_gaps"].extend(result.security_issues)
            
            # Collect feature gaps
            if result.failure_categories and 'incomplete_features' in result.failure_categories:
                patterns["feature_gaps"].append(item["task"]["expected_features"])
            
            # Collect performance issues
            if result.gas_efficiency < 0.8:
                patterns["performance_issues"].append("gas_inefficiency")
        
        return patterns
    
    def _generate_training_examples(self, training_batch: List[Dict]) -> List[Dict]:
        """Generate training examples from failures"""
        training_examples = []
        
        for item in training_batch:
            task = item["task"]
            result = item["result"]
            
            # Create preference pair for DPO training
            training_example = {
                "prompt": task["prompt"],
                "rejected_response": result.response,
                "preferred_response": self._generate_corrected_response(task, result),
                "failure_categories": result.failure_categories,
                "improvement_suggestions": result.improvement_suggestions,
                "timestamp": item["timestamp"]
            }
            
            training_examples.append(training_example)
        
        return training_examples
    
    def _generate_corrected_response(self, task: Dict, result: TestResult) -> str:
        """Generate a corrected response based on failure analysis"""
        # This is a simplified version - in practice, this would use
        # more sophisticated correction techniques
        
        corrections = []
        
        if 'compilation_failure' in result.failure_categories:
            corrections.append("// Fixed compilation errors")
        
        if 'security_issues' in result.failure_categories:
            corrections.extend([
                "import '@openzeppelin/contracts/security/ReentrancyGuard.sol';",
                "import '@openzeppelin/contracts/access/Ownable.sol';"
            ])
        
        if 'gas_inefficiency' in result.failure_categories:
            corrections.append("// Optimized for gas efficiency")
        
        # Create a basic corrected template
        corrected_response = f"""
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

{chr(10).join(corrections)}

contract {task.get('name', 'Example').replace(' ', '')} {{
    // Corrected implementation based on failure analysis
    // This would be a proper implementation addressing the specific issues
    
    constructor() {{
        // Initialization
    }}
    
    // Implementation would go here with proper:
    // - Security measures
    // - Gas optimization
    // - Feature completeness
    // - Error handling
}}
"""
        return corrected_response
    
    def _save_training_data(self, training_examples: List[Dict], failure_patterns: Dict):
        """Save training data for later use"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save training examples
        training_file = f"training_data_{timestamp}.json"
        with open(training_file, 'w') as f:
            json.dump(training_examples, f, indent=2, default=str)
        
        # Save failure patterns
        patterns_file = f"failure_patterns_{timestamp}.json"
        with open(patterns_file, 'w') as f:
            json.dump(failure_patterns, f, indent=2, default=str)
        
        print(f"📁 Training data saved: {training_file}")
        print(f"📁 Failure patterns saved: {patterns_file}")
    
    def _generate_final_analysis(self) -> Dict:
        """Generate comprehensive final analysis"""
        if not self.test_results:
            return {}
        
        # Calculate final metrics
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.overall_score > 0.8)
        
        # Performance trends
        scores = [r.overall_score for r in self.test_results]
        avg_score = sum(scores) / len(scores)
        
        # Improvement trend (if enough data)
        improvement_trend = 0.0
        if len(scores) >= 3:
            first_half = scores[:len(scores)//2]
            second_half = scores[len(scores)//2:]
            improvement_trend = sum(second_half)/len(second_half) - sum(first_half)/len(first_half)
        
        # Common failure categories
        all_failures = []
        for result in self.test_results:
            if result.failure_categories:
                all_failures.extend(result.failure_categories)
        
        failure_counts = {}
        for failure in all_failures:
            failure_counts[failure] = failure_counts.get(failure, 0) + 1
        
        analysis = {
            "session_summary": {
                "session_id": self.learning_metrics.session_id,
                "duration_minutes": (datetime.now() - self.learning_metrics.start_time).total_seconds() / 60,
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
                "average_score": avg_score,
                "improvement_trend": improvement_trend
            },
            "performance_breakdown": {
                "compilation_rate": self.learning_metrics.avg_compilation_rate,
                "security_score": self.learning_metrics.avg_security_score,
                "gas_efficiency": self.learning_metrics.avg_gas_efficiency,
                "overall_score": self.learning_metrics.avg_overall_score
            },
            "learning_progress": {
                "training_sessions": self.learning_metrics.training_sessions_triggered,
                "improvement_rate": self.learning_metrics.improvement_rate,
                "training_queue_size": self.training_queue.qsize()
            },
            "failure_analysis": {
                "common_failures": failure_counts,
                "improvement_areas": self._identify_improvement_areas(failure_counts)
            }
        }
        
        return analysis
    
    def _identify_improvement_areas(self, failure_counts: Dict) -> List[str]:
        """Identify key areas for improvement based on failure patterns"""
        areas = []
        
        if failure_counts.get('compilation_failure', 0) > 2:
            areas.append("Solidity syntax and compilation")
        
        if failure_counts.get('security_issues', 0) > 2:
            areas.append("Security best practices and vulnerability prevention")
        
        if failure_counts.get('gas_inefficiency', 0) > 2:
            areas.append("Gas optimization and efficient coding patterns")
        
        if failure_counts.get('incomplete_features', 0) > 2:
            areas.append("Feature completeness and requirement understanding")
        
        return areas
    
    def _save_results(self):
        """Save comprehensive test results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"defi_test_results_{timestamp}.json"
        
        # Prepare results for JSON serialization
        serializable_results = []
        for result in self.test_results:
            result_dict = asdict(result)
            result_dict['timestamp'] = result.timestamp.isoformat()
            serializable_results.append(result_dict)
        
        # Prepare metrics for JSON serialization
        metrics_dict = asdict(self.learning_metrics)
        metrics_dict['start_time'] = self.learning_metrics.start_time.isoformat()
        
        final_data = {
            "test_results": serializable_results,
            "learning_metrics": metrics_dict,
            "final_analysis": self._generate_final_analysis()
        }
        
        with open(results_file, 'w') as f:
            json.dump(final_data, f, indent=2, default=str)
        
        print(f"\n💾 Complete results saved to: {results_file}")
    
    def display_live_dashboard(self):
        """Display live performance dashboard"""
        if not self.test_results:
            return
        
        latest_result = self.test_results[-1]
        
        print("\n" + "="*80)
        print("🧠 CLINE-OPTIMAL DEFI DEVELOPMENT PERFORMANCE")
        print("="*80)
        print(f"Session: {self.learning_metrics.session_id}")
        print(f"Tests Completed: {self.learning_metrics.total_tests}")
        print(f"Success Rate: {(self.learning_metrics.successful_tests/self.learning_metrics.total_tests)*100:.1f}%")
        print()
        print("📊 Live Metrics:")
        print(f"├─ Compilation Success: {self.learning_metrics.avg_compilation_rate*100:.1f}%")
        print(f"├─ Security Score: {self.learning_metrics.avg_security_score*100:.1f}%")
        print(f"├─ Gas Efficiency: {self.learning_metrics.avg_gas_efficiency*100:.1f}%")
        print(f"├─ Overall Score: {self.learning_metrics.avg_overall_score*100:.1f}%")
        print(f"└─ Improvement Rate: {self.learning_metrics.improvement_rate*100:+.1f}%")
        print()
        print(f"🎯 Training Sessions: {self.learning_metrics.training_sessions_triggered}")
        print(f"📚 Training Queue: {self.training_queue.qsize()} examples")
        print("="*80)

def main():
    """Main function to run the continuous learning tester"""
    print("🚀 DeFi Continuous Learning Tester")
    print("=" * 50)
    print("This will test cline-optimal on progressive DeFi development tasks")
    print("and automatically generate training data from failures.")
    print()
    
    # Initialize tester
    tester = ContinuousLearningTester()
    
    try:
        # Run progressive testing
        results = tester.run_progressive_test(start_level=1, end_level=3)  # Start with levels 1-3
        
        # Display final dashboard
        tester.display_live_dashboard()
        
        # Print final analysis
        print("\n📊 FINAL ANALYSIS")
        print("=" * 50)
        analysis = results.get("session_summary", {})
        print(f"Total Tests: {analysis.get('total_tests', 0)}")
        print(f"Success Rate: {analysis.get('success_rate', 0)*100:.1f}%")
        print(f"Average Score: {analysis.get('average_score', 0)*100:.1f}%")
        print(f"Improvement Trend: {analysis.get('improvement_trend', 0)*100:+.1f}%")
        
        failure_analysis = results.get("failure_analysis", {})
        if failure_analysis.get("improvement_areas"):
            print("\n🎯 Key Improvement Areas:")
            for area in failure_analysis["improvement_areas"]:
                print(f"  • {area}")
        
        print("\n✅ Testing complete! Check generated files for detailed results.")
        
    except KeyboardInterrupt:
        print("\n⏹️ Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        tester.validator.cleanup()

if __name__ == "__main__":
    main()
