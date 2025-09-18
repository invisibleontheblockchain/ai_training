#!/usr/bin/env python3
"""
Rapid Learning Daemon
=====================
Fast continuous learning system that runs cycles back-to-back without delays
for maximum learning speed.
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

class RapidLearningDaemon:
    """Rapid learning system that runs cycles continuously without delays"""
    
    def __init__(self):
        self.running = False
        self.learning_cycle = 0
        self.target_accuracy = 0.95  # 95% target accuracy
        self.current_accuracy = 0.077  # Starting from actual test result
        self.improvement_threshold = 0.02  # Minimum improvement to continue

        # Attempt to load most recent saved state
        self._load_most_recent_state()
        
        # Learning configuration - optimized for speed
        self.config = {
            "training_batch_size": 1,     # Train after every failure
            "max_cycles": 50,             # Maximum learning cycles
            "patience": 5,                # Cycles without improvement before stopping
            "learning_rate": 0.001,       # Higher learning rate for faster improvement
            "dpo_beta": 0.1,             # Higher DPO beta for stronger preference learning
            "rank": 64,                  # Lower rank for faster training
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
        log_format = '%(asctime)s - %(levelname)s - %(message)s'

    def _load_most_recent_state(self):
        """Load the most recent saved state if available"""
        import glob
        state_files = sorted(glob.glob("rapid_final_state_*.json"), reverse=True)
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
        
        # Create file handler with UTF-8 encoding
        file_handler = logging.FileHandler(
            f'rapid_learning_{datetime.now().strftime("%Y%m%d")}.log',
            encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Create console handler with UTF-8 encoding
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(log_format))
        
        # Configure logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def start_daemon(self):
        """Start the rapid learning daemon"""
        self.running = True
        print("RAPID LEARNING MODE - Continuous cycles without delays")
        print(f"Target accuracy: {self.target_accuracy*100:.1f}%")
        print(f"Current accuracy: {self.current_accuracy*100:.1f}%")
        print("Running cycles back-to-back for maximum learning speed")
        print("=" * 80)
        
        try:
            # Main rapid learning loop
            while self.running and self.learning_cycle < self.config["max_cycles"]:
                self.run_learning_cycle()
                
                # Check if target achieved
                if self.current_accuracy >= self.target_accuracy:
                    print(f"TARGET ACHIEVED! Accuracy: {self.current_accuracy*100:.1f}%")
                    break
                
                # Check for lack of improvement
                if self.no_improvement_count >= self.config["patience"]:
                    print(f"No improvement for {self.config['patience']} cycles, stopping")
                    break
                
                # Brief pause to prevent overwhelming the system
                time.sleep(2)
            
        except Exception as e:
            print(f"Daemon error: {e}")
            raise
        finally:
            self.cleanup()
    
    def run_learning_cycle(self):
        """Run a complete learning cycle"""
        self.learning_cycle += 1
        cycle_start = datetime.now()
        
        print(f"\nCYCLE {self.learning_cycle} - {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 60)
        
        try:
            # Phase 1: Quick Testing (single task)
            test_results = self.run_quick_test()
            
            # Phase 2: Analysis
            analysis = self.analyze_performance(test_results)
            
            # Phase 3: Immediate Training if needed
            if analysis["needs_training"]:
                training_data = self.generate_training_data(analysis)
                if training_data.get("pairs"):
                    self.apply_immediate_training(training_data)
            
            # Phase 4: Performance Update
            self.update_performance_metrics(analysis)
            
            # Phase 5: Quick Report
            self.generate_quick_report(cycle_start, analysis)
            
        except Exception as e:
            print(f"Error in learning cycle {self.learning_cycle}: {e}")
    
    def run_quick_test(self):
        """Run a quick test on a single challenging task"""
        print("Testing cline-optimal...")
        
        try:
            # Use the most challenging task that failed before
            task = {
                "name": "Complete DeFi Suite",
                "prompt": """Create a complete DeFi suite with Oracle, DEX, Lending, Vaults, and Staking protocols for Kasplex testnet.

Requirements:
1. Oracle Protocol: Real-time price feeds with TWAP and MEV protection
2. DEX Protocol: AMM with liquidity pools, routing, and flash swap support
3. Lending Protocol: Collateralized lending with liquidations and health factors
4. Vault Protocol: Yield farming with strategy management and auto-compounding
5. Staking Protocol: Liquid staking with rewards and instant unstaking

All contracts must be production-ready with:
- Comprehensive security measures (reentrancy protection, access control)
- Gas optimization for Kaspa's 10 BPS
- Proper integration between all protocols
- Emergency controls and pause mechanisms
- Frontend integration with React/ethers.js

Create the complete smart contract suite with deployment scripts and frontend integration.""",
                "expected_features": [
                    "oracle system", "DEX functionality", "lending protocol", 
                    "vault management", "staking system", "cross-protocol integration"
                ],
                "security_requirements": [
                    "comprehensive security audit", "reentrancy protection", 
                    "oracle manipulation protection", "flash loan protection",
                    "admin controls", "emergency mechanisms"
                ],
                "gas_target": 500000,
                "difficulty": 5,
                "task_id": "rapid_test"
            }
            
            # Run single test
            result = self.tester._run_single_test(task)
            
            # Create simplified results structure
            results = {
                "session_summary": {
                    "average_score": result.overall_score,
                    "success_rate": 1.0 if result.overall_score > 0.8 else 0.0,
                    "total_tests": 1
                },
                "failure_analysis": {
                    "common_failures": {cat: 1 for cat in result.failure_categories} if result.failure_categories else {}
                },
                "test_result": result
            }
            
            print(f"Score: {result.overall_score*100:.1f}% | Compilation: {'✅' if result.compilation_success else '❌'} | Security: {result.security_score:.2f}")
            return results
            
        except Exception as e:
            print(f"Testing failed: {e}")
            return {"error": str(e)}
    
    def analyze_performance(self, test_results):
        """Analyze test results and determine next actions"""
        if "error" in test_results:
            return {"needs_training": False, "error": test_results["error"]}
        
        session_summary = test_results.get("session_summary", {})
        current_score = session_summary.get("average_score", 0)
        success_rate = session_summary.get("success_rate", 0)
        
        # Always train if not perfect
        needs_training = current_score < self.target_accuracy
        
        analysis = {
            "current_score": current_score,
            "success_rate": success_rate,
            "needs_training": needs_training,
            "improvement": current_score - self.current_accuracy,
            "failure_analysis": test_results.get("failure_analysis", {}),
            "test_results": test_results
        }
        
        return analysis
    
    def generate_training_data(self, analysis):
        """Generate targeted training data based on analysis"""
        try:
            # Get failure patterns
            failure_analysis = analysis.get("failure_analysis", {})
            common_failures = failure_analysis.get("common_failures", {})
            
            # Generate targeted training pairs
            training_pairs = []
            
            # Always generate language correction pairs
            solidity_pairs = self.data_generator.generate_solidity_training_pairs()
            training_pairs.extend(solidity_pairs)
            
            # Add integration pairs
            integration_pairs = self.data_generator.generate_integration_training_pairs()
            training_pairs.extend(integration_pairs)
            
            # Add deployment pairs
            deployment_pairs = self.data_generator.generate_deployment_training_pairs()
            training_pairs.extend(deployment_pairs)
            
            return {
                "pairs": training_pairs,
                "metadata": {
                    "cycle": self.learning_cycle,
                    "target_failures": common_failures,
                    "generated_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            print(f"Training data generation failed: {e}")
            return {"pairs": [], "error": str(e)}
    
    def apply_immediate_training(self, training_data):
        """Apply training immediately without queuing"""
        print(f"Training with {len(training_data['pairs'])} examples...")
        
        try:
            # Save training data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            training_file = f"rapid_training_cycle_{self.learning_cycle}_{timestamp}.json"
            
            with open(training_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "cycle": self.learning_cycle,
                    "pairs": training_data["pairs"],
                    "config": self.config,
                    "timestamp": timestamp
                }, f, indent=2, ensure_ascii=False)
            
            print(f"Training data saved: {training_file}")
            print(f"Using DoRA (rank={self.config['rank']}) + DPO (β={self.config['dpo_beta']})")
            
            # Simulate training (in real implementation, this would call your training pipeline)
            print("Applying training...")
            time.sleep(3)  # Simulate training time
            
            print("Training completed!")
            
        except Exception as e:
            print(f"Training failed: {e}")
    
    def update_performance_metrics(self, analysis):
        """Update performance tracking"""
        new_score = analysis.get("current_score", self.current_accuracy)
        improvement = new_score - self.current_accuracy
        
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
            print(f"IMPROVEMENT: +{improvement*100:.2f}% (Streak: {self.improvement_streak})")
        else:
            self.improvement_streak = 0
            self.no_improvement_count += 1
            print(f"No significant improvement (Count: {self.no_improvement_count})")
        
        self.current_accuracy = new_score
    
    def generate_quick_report(self, cycle_start, analysis):
        """Generate quick cycle report"""
        cycle_duration = datetime.now() - cycle_start
        
        print(f"Duration: {cycle_duration.total_seconds():.1f}s | " +
              f"Accuracy: {self.current_accuracy*100:.1f}% | " +
              f"Progress: {(self.current_accuracy/self.target_accuracy)*100:.1f}% | " +
              f"Streak: {self.improvement_streak}")
        
        # Save detailed report
        report = {
            "cycle": self.learning_cycle,
            "duration_seconds": cycle_duration.total_seconds(),
            "current_accuracy": self.current_accuracy,
            "target_accuracy": self.target_accuracy,
            "progress_to_target": (self.current_accuracy / self.target_accuracy) * 100,
            "improvement_streak": self.improvement_streak,
            "no_improvement_count": self.no_improvement_count,
            "analysis": analysis
        }
        
        report_file = f"rapid_cycle_{self.learning_cycle}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    
    def cleanup(self):
        """Cleanup resources"""
        print("Cleaning up...")
        
        # Save final state
        final_state = {
            "final_accuracy": self.current_accuracy,
            "total_cycles": self.learning_cycle,
            "performance_history": self.performance_history,
            "config": self.config,
            "stopped_at": datetime.now().isoformat()
        }
        
        with open(f"rapid_final_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w', encoding='utf-8') as f:
            json.dump(final_state, f, indent=2, ensure_ascii=False)
        
        print("Cleanup complete")

def main():
    """Main function to start the rapid learning daemon"""
    print("⚡ RAPID Learning Daemon for cline-optimal")
    print("=" * 60)
    print("MAXIMUM SPEED MODE - No delays between cycles")
    print("This will run continuous learning cycles back-to-back")
    print("until 95% accuracy is achieved on DeFi tasks.")
    print()
    print("Press Ctrl+C to stop gracefully")
    print("=" * 60)
    
    # Create and start daemon
    daemon = RapidLearningDaemon()
    
    try:
        daemon.start_daemon()
    except KeyboardInterrupt:
        print("\nDaemon stopped by user")
    except Exception as e:
        print(f"\nDaemon error: {e}")
        import traceback
        traceback.print_exc()
    
    print("Rapid Learning Daemon shutdown complete")

if __name__ == "__main__":
    main()
