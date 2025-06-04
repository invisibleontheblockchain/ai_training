"""
Model Comparison Runner
======================
Easy-to-use interface for running different types of model comparisons.

Usage:
  python run_comparison.py --quick          # Quick basic tests
  python run_comparison.py --full           # Full comprehensive tests  
  python run_comparison.py --speed          # Speed-only benchmarks
  python run_comparison.py --custom         # Custom test selection

Created: June 4, 2025
"""

import argparse
import sys
import os
import json
from pathlib import Path
from comprehensive_model_comparison import ComprehensiveModelComparison, TestConfig

def run_quick_comparison():
    """Run a quick comparison with basic tests only"""
    print("🚀 Running Quick Comparison (Basic Tests Only)")
    print("=" * 60)
    
    config = TestConfig()
    config.test_iterations = 3  # Fewer iterations for speed
    config.warmup_iterations = 1
    
    comparison = ComprehensiveModelComparison(config)
    results = comparison.run_comprehensive_comparison(["basic"])
    
    comparison.print_live_summary()
    return results

def run_speed_only():
    """Run speed benchmarks only"""
    print("⚡ Running Speed-Only Benchmarks")
    print("=" * 60)
    
    config = TestConfig()
    comparison = ComprehensiveModelComparison(config)
    
    # Setup models
    phi2_ready = comparison.setup_phi2_model()
    cline_ready = comparison.setup_cline_optimal()
    
    if not cline_ready:
        print("❌ Cline-optimal model not available")
        return None
    
    results = {}
    
    # Run speed tests with different iterations
    for iterations in [5, 10, 20]:
        print(f"\n🔄 Testing with {iterations} iterations...")
        
        if phi2_ready:
            phi2_speed = comparison.run_speed_benchmark("phi2_finetuned", iterations)
            results[f"phi2_speed_{iterations}"] = phi2_speed
            print(f"  Phi-2: {phi2_speed.get('avg_time', 'N/A'):.3f}s avg")
        
        if cline_ready:
            cline_speed = comparison.run_speed_benchmark("cline_optimal", iterations)
            results[f"cline_speed_{iterations}"] = cline_speed
            print(f"  Cline: {cline_speed.get('avg_time', 'N/A'):.3f}s avg")
    
    # Save speed results
    timestamp = comparison.results["timestamp"]
    speed_file = f"{config.results_dir}/speed_only_{timestamp}.json"
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)
    
    with open(speed_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Speed results saved to: {speed_file}")
    
    return results

def run_full_comparison():
    """Run the full comprehensive comparison"""
    print("🎯 Running Full Comprehensive Comparison")
    print("=" * 60)
    
    config = TestConfig()
    comparison = ComprehensiveModelComparison(config)
    results = comparison.run_comprehensive_comparison()
    
    comparison.print_live_summary()
    return results

def run_custom_comparison():
    """Run a custom comparison with user-selected options"""
    print("🛠️ Custom Comparison Setup")
    print("=" * 60)
    
    # Test level selection
    available_levels = ["basic", "intermediate", "advanced", "autonomous"]
    print("\nAvailable test levels:")
    for i, level in enumerate(available_levels, 1):
        print(f"  {i}. {level}")
    
    selected = input("\nEnter test levels (comma-separated numbers or names): ").strip()
    
    # Parse selection
    test_levels = []
    if selected:
        for item in selected.split(","):
            item = item.strip()
            if item.isdigit():
                idx = int(item) - 1
                if 0 <= idx < len(available_levels):
                    test_levels.append(available_levels[idx])
            elif item in available_levels:
                test_levels.append(item)
    
    if not test_levels:
        test_levels = available_levels
    
    # Configuration options
    print(f"\nSelected test levels: {', '.join(test_levels)}")
    
    iterations = input("Number of test iterations per test (default 5): ").strip()
    if iterations.isdigit():
        iterations = int(iterations)
    else:
        iterations = 5
    
    warmup = input("Number of warmup iterations (default 2): ").strip()
    if warmup.isdigit():
        warmup = int(warmup)
    else:
        warmup = 2
    
    max_tokens = input("Max tokens per response (default 512): ").strip()
    if max_tokens.isdigit():
        max_tokens = int(max_tokens)
    else:
        max_tokens = 512
    
    # Create custom config
    config = TestConfig()
    config.test_iterations = iterations
    config.warmup_iterations = warmup
    config.max_new_tokens = max_tokens
    
    print(f"\n🚀 Running custom comparison...")
    print(f"  Test levels: {', '.join(test_levels)}")
    print(f"  Iterations: {iterations}")
    print(f"  Warmup: {warmup}")
    print(f"  Max tokens: {max_tokens}")
    print("=" * 60)
    
    comparison = ComprehensiveModelComparison(config)
    results = comparison.run_comprehensive_comparison(test_levels)
    
    comparison.print_live_summary()
    return results

def run_stress_test():
    """Run a stress test with many iterations"""
    print("💪 Running Stress Test")
    print("=" * 60)
    
    config = TestConfig()
    config.test_iterations = 10
    config.warmup_iterations = 3
    
    comparison = ComprehensiveModelComparison(config)
    
    # Focus on basic tests but with many iterations
    print("Running stress test with basic prompts and high iteration count...")
    results = comparison.run_comprehensive_comparison(["basic"])
    
    # Also run extended speed tests
    print("\n⚡ Extended speed testing...")
    phi2_ready = hasattr(comparison, 'phi2_pipeline')
    cline_ready = True  # Assume cline is available if we got this far
    
    if phi2_ready:
        phi2_stress = comparison.run_speed_benchmark("phi2_finetuned", 50)
        print(f"Phi-2 stress test (50 iterations): {phi2_stress.get('avg_time', 'N/A'):.3f}s avg")
    
    if cline_ready:
        cline_stress = comparison.run_speed_benchmark("cline_optimal", 50)
        print(f"Cline stress test (50 iterations): {cline_stress.get('avg_time', 'N/A'):.3f}s avg")
    
    comparison.print_live_summary()
    return results

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="Model Comparison Runner")
    parser.add_argument("--quick", action="store_true", help="Run quick basic comparison")
    parser.add_argument("--full", action="store_true", help="Run full comprehensive comparison")
    parser.add_argument("--speed", action="store_true", help="Run speed-only benchmarks")
    parser.add_argument("--custom", action="store_true", help="Run custom comparison")
    parser.add_argument("--stress", action="store_true", help="Run stress test")
    parser.add_argument("--setup", action="store_true", help="Run setup first")
    
    args = parser.parse_args()
    
    # If setup requested, run setup first
    if args.setup:
        print("🔧 Running setup first...")
        try:
            from setup_comparison import main as setup_main
            if not setup_main():
                print("❌ Setup failed")
                return 1
        except ImportError:
            print("❌ setup_comparison.py not found")
            return 1
        print("✅ Setup complete, proceeding with comparison...\n")
    
    # Check if any specific mode was requested
    if args.quick:
        results = run_quick_comparison()
    elif args.full:
        results = run_full_comparison()
    elif args.speed:
        results = run_speed_only()
    elif args.custom:
        results = run_custom_comparison()
    elif args.stress:
        results = run_stress_test()
    else:
        # No specific mode, show interactive menu
        print("🤖 Model Comparison Tool")
        print("=" * 30)
        print("1. Quick comparison (basic tests)")
        print("2. Full comparison (all test levels)")
        print("3. Speed-only benchmarks")
        print("4. Custom comparison")
        print("5. Stress test")
        print("6. Setup and then compare")
        print("0. Exit")
        
        while True:
            choice = input("\nSelect option (0-6): ").strip()
            
            if choice == "0":
                print("Goodbye!")
                return 0
            elif choice == "1":
                results = run_quick_comparison()
                break
            elif choice == "2":
                results = run_full_comparison()
                break
            elif choice == "3":
                results = run_speed_only()
                break
            elif choice == "4":
                results = run_custom_comparison()
                break
            elif choice == "5":
                results = run_stress_test()
                break
            elif choice == "6":
                try:
                    from setup_comparison import main as setup_main
                    if setup_main():
                        print("\n" + "="*60)
                        results = run_full_comparison()
                    else:
                        print("❌ Setup failed")
                        return 1
                    break
                except ImportError:
                    print("❌ setup_comparison.py not found")
                    return 1
            else:
                print("Invalid choice, please select 0-6")
    
    if results:
        print(f"\n🎉 Comparison complete!")
        return 0
    else:
        print(f"\n❌ Comparison failed or was cancelled")
        return 1

if __name__ == "__main__":
    sys.exit(main())
