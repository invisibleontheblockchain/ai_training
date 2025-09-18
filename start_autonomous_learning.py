#!/usr/bin/env python3
"""
Autonomous Learning Launcher
============================
Simple launcher for the autonomous learning daemon with configuration options.
"""

import argparse
import sys
import os
from pathlib import Path

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description="Start the Autonomous Learning Daemon for cline-optimal")
    
    # Configuration options
    parser.add_argument("--target-accuracy", type=float, default=0.95, 
                       help="Target accuracy to achieve (default: 0.95)")
    parser.add_argument("--test-interval", type=int, default=30,
                       help="Minutes between test cycles (default: 30)")
    parser.add_argument("--max-cycles", type=int, default=100,
                       help="Maximum learning cycles (default: 100)")
    parser.add_argument("--quick-test", action="store_true",
                       help="Run with shorter intervals for testing (5 min cycles)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    print("🚀 Autonomous Learning System for cline-optimal")
    print("=" * 60)
    print(f"Target Accuracy: {args.target_accuracy*100:.1f}%")
    print(f"Test Interval: {args.test_interval} minutes")
    print(f"Max Cycles: {args.max_cycles}")
    print(f"Quick Test Mode: {args.quick_test}")
    print("=" * 60)
    
    # Import and configure the daemon
    try:
        from autonomous_learning_daemon import AutonomousLearningDaemon
        
        # Create daemon with custom configuration
        daemon = AutonomousLearningDaemon()
        
        # Apply command line arguments
        daemon.target_accuracy = args.target_accuracy
        daemon.config["max_cycles"] = args.max_cycles
        
        if args.quick_test:
            daemon.config["test_interval_minutes"] = 5  # Quick testing
            daemon.config["training_batch_size"] = 2   # Smaller batches
            print("⚡ Quick test mode enabled - 5 minute cycles")
        else:
            daemon.config["test_interval_minutes"] = args.test_interval
        
        if args.verbose:
            import logging
            logging.getLogger().setLevel(logging.DEBUG)
            print("🔍 Verbose logging enabled")
        
        print("\n🎯 LEARNING OBJECTIVES:")
        print(f"  • Current baseline: 11.9% (from testing)")
        print(f"  • Target accuracy: {args.target_accuracy*100:.1f}%")
        print(f"  • Improvement needed: +{(args.target_accuracy - 0.119)*100:.1f}%")
        print(f"  • Key focus areas:")
        print(f"    - Generate Solidity instead of Python")
        print(f"    - Add comprehensive security patterns")
        print(f"    - Include frontend integration")
        print(f"    - Provide deployment scripts")
        
        print(f"\n⏰ SCHEDULE:")
        print(f"  • Test every {daemon.config['test_interval_minutes']} minutes")
        print(f"  • Train after every 5 failures")
        print(f"  • Maximum {args.max_cycles} learning cycles")
        print(f"  • Stop after 3 cycles without improvement")
        
        print(f"\n📊 MONITORING:")
        print(f"  • Real-time performance tracking")
        print(f"  • Automatic training data generation")
        print(f"  • Cycle reports and progress logs")
        print(f"  • Graceful shutdown with Ctrl+C")
        
        input(f"\nPress Enter to start the autonomous learning system...")
        
        # Start the daemon
        daemon.start_daemon()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required modules are available")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ Startup cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
