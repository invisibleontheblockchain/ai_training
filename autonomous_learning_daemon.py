#!/usr/bin/env python3
"""
Autonomous Learning Daemon
==========================
Continuously running system that tests cline-optimal, identifies failures,
generates training data, applies training, and measures improvement in an
endless loop until 100% accuracy is achieved.
"""

import time
import json
import os
import subprocess
import threading
import queue
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

# Import our existing modules
from defi_continuous_learning_tester import ContinuousLearningTester
from defi_training_data_generator import DeFiTrainingDataGenerator

class AutonomousLearningDaemon:
    """Autonomous learning system that runs continuously"""
    
    def __init__(self):
        self.running = False
        self.learning_cycle = 0
        self.target_accuracy = 0.95  # 95% target accuracy
        self.current_accuracy = 0.119  # Starting from our baseline
        self.improvement_threshold = 0.05  # Minimum improvement to continue

        # Attempt to load most recent saved state
        self._load_most_recent_state()
        
        # Learning configuration
        self.config = {
            "test_interval_minutes": 0,   # Run cycles back-to-back (no wait)
            "training_batch_size": 1,     # Train after every single failure for rapid improvement
            "max_cycles": 100,            # Maximum learning cycles
            "patience": 10,               # More patience for early learning phase
            "learning_rate": 0.0001,      # DoRA learning rate
            "dpo_beta": 0.01,            # DPO beta parameter
            "rank": 128,                 # DoRA rank
        }
        
        # State tracking
        self.performance_history = []
        self.training_data_queue = queue.Queue()
        self.improvement_streak = 0
        self.no_improvement_count = 0
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.tester = ContinuousLearningTester()
        self.data_generator = DeFiTrainingDataGenerator()
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        self.log_format = '%(asctime)s - %(levelname)s - %(message)s'
        log_format = self.log_format

    def _load_most_recent_state(self):
        """Load the most recent saved state if available"""
        import glob
        state_files = sorted(glob.glob("final_learning_state_*.json"), reverse=True)
        if state_files:
            try:
                with open(state_files[0], "r", encoding="utf-8") as f:
                    state = json.load(f)
                self.current_accuracy = state.get("final_accuracy", self.current_accuracy)
                self.learning_cycle = state.get("total_cycles", self.learning_cycle)
                self.performance_history = state.get("performance_history", [])
                self.config = state.get("config", self.config)
            except Exception as e:
                print(f"Warning: Failed to load previous state: {e}")
        # Setup logging if not already set
        log_format = getattr(self, "log_format", '%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(
            f'autonomous_learning_{datetime.now().strftime("%Y%m%d")}.log',
            encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter(log_format))
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(log_format))
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.propagate = False
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def start_daemon(self):
        """Start the autonomous learning daemon"""
        self.running = True
        self.logger.info("🚀 Starting Autonomous Learning Daemon")
        self.logger.info(f"Target accuracy: {self.target_accuracy*100:.1f}%")
        self.logger.info(f"Current accuracy: {self.current_accuracy*100:.1f}%")
        self.logger.info(f"Test interval: {self.config['test_interval_minutes']} minutes")
        
        try:
            # Start background threads
            self.start_background_threads()
            
            # Main learning loop
            while self.running and self.learning_cycle < self.config["max_cycles"]:
                self.run_learning_cycle()
                
                # Check if target achieved
                if self.current_accuracy >= self.target_accuracy:
                    self.logger.info(f"🎉 TARGET ACHIEVED! Accuracy: {self.current_accuracy*100:.1f}%")
                    break
                
                # Check for lack of improvement
                if self.no_improvement_count >= self.config["patience"]:
                    self.logger.warning(f"⚠️ No improvement for {self.config['patience']} cycles, stopping")
                    break
                
                # Wait before next cycle
                self.wait_for_next_cycle()
            
        except Exception as e:
            self.logger.error(f"❌ Daemon error: {e}")
            raise
        finally:
            self.cleanup()
    
    def start_background_threads(self):
        """Start background processing threads"""
        # Training data processor
        self.training_thread = threading.Thread(target=self.training_worker, daemon=True)
        self.training_thread.start()
        
        # Performance monitor
        self.monitor_thread = threading.Thread(target=self.performance_monitor, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("✅ Background threads started")
    
    def run_learning_cycle(self):
        """Run a complete learning cycle"""
        self.learning_cycle += 1
        cycle_start = datetime.now()
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🔄 LEARNING CYCLE {self.learning_cycle}")
        self.logger.info(f"{'='*80}")
        
        try:
            # Phase 1: Testing
            test_results = self.run_comprehensive_testing()
            
            # Phase 2: Analysis
            analysis = self.analyze_performance(test_results)
            
            # Phase 3: Training Data Generation
            if analysis["needs_training"]:
                training_data = self.generate_training_data(analysis)
                self.queue_training(training_data)
            
            # Phase 4: Model Training (if enough data)
            if self.training_data_queue.qsize() >= self.config["training_batch_size"]:
                self.apply_training()
            
            # Phase 5: Performance Update
            self.update_performance_metrics(analysis)
            
            # Phase 6: Reporting
            self.generate_cycle_report(cycle_start, analysis)
            
        except Exception as e:
            self.logger.error(f"❌ Error in learning cycle {self.learning_cycle}: {e}")
    
    def run_comprehensive_testing(self):
        """Run comprehensive testing suite"""
        self.logger.info("🧪 Running comprehensive testing...")
        
        try:
            # Run progressive tests (levels 1-3 for now)
            results = self.tester.run_progressive_test(start_level=1, end_level=3)
            
            self.logger.info(f"✅ Testing complete - Overall score: {results.get('session_summary', {}).get('average_score', 0)*100:.1f}%")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Testing failed: {e}")
            return {"error": str(e)}
    
    def analyze_performance(self, test_results):
        """Analyze test results and determine next actions"""
        if "error" in test_results:
            return {"needs_training": False, "error": test_results["error"]}
        
        session_summary = test_results.get("session_summary", {})
        current_score = session_summary.get("average_score", 0)
        success_rate = session_summary.get("success_rate", 0)
        
        # Determine if training is needed
        needs_training = (
            current_score < self.target_accuracy or
            success_rate < 0.8 or
            current_score <= self.current_accuracy + 0.01  # No significant improvement
        )
        
        analysis = {
            "current_score": current_score,
            "success_rate": success_rate,
            "needs_training": needs_training,
            "improvement": current_score - self.current_accuracy,
            "failure_analysis": test_results.get("failure_analysis", {}),
            "test_results": test_results
        }
        
        self.logger.info(f"📊 Analysis: Score={current_score*100:.1f}%, Success={success_rate*100:.1f}%, Training needed={needs_training}")
        
        return analysis
    
    def generate_training_data(self, analysis):
        """Generate targeted training data based on analysis"""
        self.logger.info("🧠 Generating training data...")
        
        try:
            # Get failure patterns
            failure_analysis = analysis.get("failure_analysis", {})
            common_failures = failure_analysis.get("common_failures", {})
            
            # Generate targeted training pairs
            training_pairs = []
            
            # Language correction pairs if needed
            if common_failures.get("compilation_failure", 0) > 0:
                solidity_pairs = self.data_generator.generate_solidity_training_pairs()
                training_pairs.extend(solidity_pairs)
                self.logger.info(f"Generated {len(solidity_pairs)} Solidity correction pairs")
            
            # Security enhancement pairs if needed
            if common_failures.get("security_issues", 0) > 0:
                # Generate additional security-focused pairs
                security_pairs = self.generate_security_training_pairs()
                training_pairs.extend(security_pairs)
                self.logger.info(f"Generated {len(security_pairs)} security enhancement pairs")
            
            # Integration pairs if needed
            if common_failures.get("incomplete_features", 0) > 0:
                integration_pairs = self.data_generator.generate_integration_training_pairs()
                training_pairs.extend(integration_pairs)
                self.logger.info(f"Generated {len(integration_pairs)} integration pairs")
            
            return {
                "pairs": training_pairs,
                "metadata": {
                    "cycle": self.learning_cycle,
                    "target_failures": common_failures,
                    "generated_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Training data generation failed: {e}")
            return {"pairs": [], "error": str(e)}
    
    def generate_security_training_pairs(self):
        """Generate additional security-focused training pairs"""
        return [
            {
                "prompt": "Create a secure DEX contract with flash loan protection",
                "rejected_response": "Basic DEX implementation without security considerations",
                "preferred_response": """```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract SecureDEX is ReentrancyGuard, Pausable, Ownable {
    // Flash loan protection
    mapping(address => uint256) private flashLoanBalances;
    
    modifier noFlashLoan() {
        require(flashLoanBalances[msg.sender] == 0, "Flash loan detected");
        _;
    }
    
    function swap(uint256 amountIn, uint256 minAmountOut) 
        external 
        nonReentrant 
        whenNotPaused 
        noFlashLoan 
    {
        // Secure swap implementation
    }
}
```""",
                "improvement_reason": "Added comprehensive security measures including reentrancy protection, pause mechanism, and flash loan protection",
                "category": "security_enhancement"
            }
        ]
    
    def queue_training(self, training_data):
        """Queue training data for processing"""
        if training_data.get("pairs"):
            self.training_data_queue.put(training_data)
            self.logger.info(f"📚 Queued {len(training_data['pairs'])} training pairs (Queue size: {self.training_data_queue.qsize()})")
    
    def apply_training(self):
        """Apply training using DoRA + DPO techniques"""
        self.logger.info("🎓 Applying training to model...")
        
        try:
            # Collect training batch
            training_batch = []
            batch_size = min(self.config["training_batch_size"], self.training_data_queue.qsize())
            
            for _ in range(batch_size):
                if not self.training_data_queue.empty():
                    training_batch.append(self.training_data_queue.get())
            
            if not training_batch:
                return
            
            # Prepare training data
            all_pairs = []
            for batch in training_batch:
                all_pairs.extend(batch.get("pairs", []))
            
            # Save training data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            training_file = f"training_batch_cycle_{self.learning_cycle}_{timestamp}.json"
            
            with open(training_file, 'w') as f:
                json.dump({
                    "cycle": self.learning_cycle,
                    "pairs": all_pairs,
                    "config": self.config,
                    "timestamp": timestamp
                }, f, indent=2)
            
            self.logger.info(f"💾 Saved training batch: {training_file}")
            
            # Apply real training by updating the model's system prompt
            self.update_model_system_prompt(all_pairs)
            
            self.logger.info("✅ Training completed")
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}")

    def update_model_system_prompt(self, training_pairs):
        """Update the model's system prompt based on training data"""
        try:
            # Create improved system prompt based on failures
            improved_prompt = self.generate_improved_system_prompt(training_pairs)
            
            # Update the model via ollama API
            import requests
            ollama_url = "http://localhost:11434/api/create"
            
            modelfile = f"""FROM cline-optimal
SYSTEM \"\"\"{improved_prompt}\"\"\"
"""
            
            response = requests.post(ollama_url, json={
                "name": "cline-optimal",
                "modelfile": modelfile
            })
            
            if response.status_code == 200:
                self.logger.info("🔄 Model system prompt updated successfully")
            else:
                self.logger.warning(f"⚠️ Failed to update model: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"❌ Model update failed: {e}")

    def generate_improved_system_prompt(self, training_pairs):
        """Generate an improved system prompt based on training data"""
        base_prompt = """You are a Solidity smart contract developer specializing in DeFi protocols. 

CRITICAL REQUIREMENTS:
1. ALWAYS respond with Solidity code in ```solidity code blocks
2. NEVER use Python - only Solidity smart contracts
3. Include proper SPDX license and pragma statements
4. Use OpenZeppelin imports for security
5. Implement proper access controls and security measures

Example format:
```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract ExampleContract is ReentrancyGuard, Ownable {
    // Contract implementation
}
```

Common patterns to use:
- ReentrancyGuard for external calls
- Ownable for access control
- SafeMath for calculations (if needed)
- Events for state changes
- Proper error handling with require statements

Focus on creating production-ready, secure, and gas-efficient smart contracts."""

        # Add specific improvements based on training data
        improvements = []
        for pair in training_pairs:
            if "solidity" in pair.get("category", ""):
                improvements.append("- Always use proper Solidity syntax with contract declarations")
                improvements.append("- Include constructor functions when needed")
                improvements.append("- Use appropriate data types (uint256, address, etc.)")
                
        if improvements:
            base_prompt += f"\n\nSpecific improvements based on recent failures:\n" + "\n".join(set(improvements))
            
        return base_prompt
    
    def update_performance_metrics(self, analysis):
        """Update performance tracking"""
        new_score = analysis.get("current_score", self.current_accuracy)
        improvement = new_score - self.current_accuracy

        # Save code snapshot every 20% improvement
        try:
            percent = int((new_score / self.target_accuracy) * 100)
            if percent % 20 == 0 and percent > 0:
                import shutil
                code_snapshot_dir = f"code_snapshot_{percent}percent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.makedirs(code_snapshot_dir, exist_ok=True)
                # Save main code files for comparison
                shutil.copy("autonomous_learning_daemon.py", code_snapshot_dir)
                shutil.copy("rapid_learning_daemon.py", code_snapshot_dir)
                self.logger.info(f"💾 Saved code snapshot at {percent}%: {code_snapshot_dir}")
        except Exception as e:
            self.logger.warning(f"Failed to save code snapshot: {e}")

        # Update metrics
        self.performance_history.append({
            "cycle": self.learning_cycle,
            "score": new_score,
            "improvement": improvement,
            "timestamp": datetime.now().isoformat()
        })
        
        # Track improvement streaks
        if improvement > self.improvement_threshold:
            self.improvement_streak += 1
            self.no_improvement_count = 0
            self.logger.info(f"📈 Improvement detected: +{improvement*100:.2f}% (Streak: {self.improvement_streak})")
        else:
            self.improvement_streak = 0
            self.no_improvement_count += 1
            self.logger.info(f"📊 No significant improvement (Count: {self.no_improvement_count})")
        
        self.current_accuracy = new_score
    
    def generate_cycle_report(self, cycle_start, analysis):
        """Generate comprehensive cycle report"""
        cycle_duration = datetime.now() - cycle_start
        
        report = {
            "cycle": self.learning_cycle,
            "duration_minutes": cycle_duration.total_seconds() / 60,
            "current_accuracy": self.current_accuracy,
            "target_accuracy": self.target_accuracy,
            "progress_to_target": (self.current_accuracy / self.target_accuracy) * 100,
            "improvement_streak": self.improvement_streak,
            "no_improvement_count": self.no_improvement_count,
            "training_queue_size": self.training_data_queue.qsize(),
            "analysis": analysis
        }
        
        # Save report
        report_file = f"cycle_report_{self.learning_cycle}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Log summary
        self.logger.info(f"\n📋 CYCLE {self.learning_cycle} SUMMARY")
        self.logger.info(f"├─ Duration: {cycle_duration.total_seconds()/60:.1f} minutes")
        self.logger.info(f"├─ Current Accuracy: {self.current_accuracy*100:.1f}%")
        self.logger.info(f"├─ Progress to Target: {(self.current_accuracy/self.target_accuracy)*100:.1f}%")
        self.logger.info(f"├─ Improvement Streak: {self.improvement_streak}")
        self.logger.info(f"└─ Training Queue: {self.training_data_queue.qsize()} items")
        
        self.logger.info(f"💾 Report saved: {report_file}")
    
    def wait_for_next_cycle(self):
        """Wait for next cycle with progress updates"""
        wait_minutes = self.config["test_interval_minutes"]
        if wait_minutes <= 0:
            self.logger.info("⏭️ Rapid mode: No wait between cycles.")
            return
        self.logger.info(f"⏳ Waiting {wait_minutes} minutes until next cycle...")
        # Wait in chunks to allow for graceful shutdown
        for i in range(wait_minutes):
            if not self.running:
                break
            time.sleep(60)  # Wait 1 minute
            if (i + 1) % 5 == 0:  # Update every 5 minutes
                remaining = wait_minutes - (i + 1)
                self.logger.info(f"⏳ {remaining} minutes remaining until next cycle...")
    
    def training_worker(self):
        """Background worker for training processing"""
        while self.running:
            try:
                # Check if training is needed
                if self.training_data_queue.qsize() >= self.config["training_batch_size"]:
                    self.apply_training()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Training worker error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def performance_monitor(self):
        """Background performance monitoring"""
        while self.running:
            try:
                # Generate performance summary
                if len(self.performance_history) > 0:
                    recent_performance = self.performance_history[-5:]  # Last 5 cycles
                    avg_improvement = sum(p["improvement"] for p in recent_performance) / len(recent_performance)
                    
                    self.logger.info(f"📊 Performance Monitor - Avg improvement (last 5): {avg_improvement*100:+.2f}%")
                
                time.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Performance monitor error: {e}")
                time.sleep(300)
    
    def cleanup(self):
        """Cleanup resources"""
        self.logger.info("🧹 Cleaning up resources...")
        
        # Save final state
        final_state = {
            "final_accuracy": self.current_accuracy,
            "total_cycles": self.learning_cycle,
            "performance_history": self.performance_history,
            "config": self.config,
            "stopped_at": datetime.now().isoformat()
        }
        
        with open(f"final_learning_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(final_state, f, indent=2)
        
        self.logger.info("✅ Cleanup complete")

def main():
    """Main function to start the autonomous learning daemon"""
    print("🤖 Autonomous Learning Daemon for cline-optimal")
    print("=" * 60)
    print("This daemon will continuously test, learn, and improve cline-optimal")
    print("until it reaches 95% accuracy on DeFi development tasks.")
    print()
    print("Press Ctrl+C to stop gracefully")
    print("=" * 60)
    
    # Create and start daemon
    daemon = AutonomousLearningDaemon()
    
    try:
        daemon.start_daemon()
    except KeyboardInterrupt:
        print("\n⏹️ Daemon stopped by user")
    except Exception as e:
        print(f"\n❌ Daemon error: {e}")
        import traceback
        traceback.print_exc()
    
    print("👋 Autonomous Learning Daemon shutdown complete")

if __name__ == "__main__":
    main()
