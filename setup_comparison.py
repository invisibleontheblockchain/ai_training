"""
Setup Script for Cline-Optimal Model Comparison
===============================================
This script helps set up the cline-optimal model and prepare for comparison testing.
It integrates with your existing PowerShell setup script.

Created: June 4, 2025
"""

import subprocess
import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, Optional

def run_powershell_command(command: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a PowerShell command and return the result"""
    try:
        result = subprocess.run(
            ["powershell.exe", "-Command", command],
            capture_output=True,
            text=True,
            check=check
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"PowerShell command failed: {e}")
        print(f"Error output: {e.stderr}")
        raise

def check_ollama_installation() -> bool:
    """Check if Ollama is installed and running"""
    try:
        result = run_powershell_command("ollama --version", check=False)
        if result.returncode == 0:
            print("✓ Ollama is installed")
            return True
        else:
            print("✗ Ollama is not installed or not in PATH")
            return False
    except Exception:
        print("✗ Ollama is not installed or not in PATH")
        return False

def check_cline_optimal_model() -> bool:
    """Check if cline-optimal model is available"""
    try:
        result = run_powershell_command("ollama list", check=False)
        if result.returncode == 0 and "cline-optimal:latest" in result.stdout:
            print("✓ cline-optimal:latest model found")
            return True
        else:
            print("✗ cline-optimal:latest model not found")
            return False
    except Exception:
        print("✗ Error checking for cline-optimal model")
        return False

def run_cline_optimal_setup(skip_model_download: bool = False) -> bool:
    """Run the cline-optimal setup PowerShell script"""
    print("\n🔧 Running cline-optimal setup...")
    
    # Find the PowerShell script
    script_paths = [
        "c:/AI_Training/models/base/rtx3090_cline_optimal_system.ps1",
        "c:/AI_Training/setup_cline_optimal.ps1",
        "./setup_cline_optimal.ps1"
    ]
    
    script_path = None
    for path in script_paths:
        if os.path.exists(path):
            script_path = path
            break
    
    if not script_path:
        print("✗ Could not find cline-optimal setup script")
        print("Expected locations:")
        for path in script_paths:
            print(f"  - {path}")
        return False
    
    print(f"✓ Found setup script: {script_path}")
    
    # Prepare PowerShell command
    ps_command = f'& "{script_path}" -Action "full"'
    if skip_model_download:
        ps_command += " -SkipModelDownload"
    
    try:
        print("Running PowerShell setup script...")
        print("This may take several minutes...")
        
        result = run_powershell_command(ps_command, check=False)
        
        if result.returncode == 0:
            print("✓ Cline-optimal setup completed successfully")
            return True
        else:
            print("✗ Cline-optimal setup failed")
            print(f"Error output: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Error running setup script: {e}")
        return False

def test_model_functionality() -> Dict[str, any]:
    """Test both models to ensure they're working"""
    print("\n🧪 Testing model functionality...")
    
    test_results = {
        "phi2_available": False,
        "cline_optimal_available": False,
        "test_timestamp": time.time()
    }
    
    # Test Phi-2 model
    phi2_path = "c:/AI_Training/models/fine-tuned/phi2-optimized"
    if os.path.exists(phi2_path):
        test_results["phi2_available"] = True
        print("✓ Phi-2 fine-tuned model found")
    else:
        print("✗ Phi-2 fine-tuned model not found")
    
    # Test cline-optimal
    if check_cline_optimal_model():
        # Try a simple test
        try:
            result = run_powershell_command(
                'ollama run cline-optimal:latest "Hello, test response"',
                check=False
            )
            if result.returncode == 0:
                test_results["cline_optimal_available"] = True
                print("✓ Cline-optimal model responding correctly")
            else:
                print("✗ Cline-optimal model not responding")
        except Exception:
            print("✗ Error testing cline-optimal model")
    
    return test_results

def install_python_dependencies() -> bool:
    """Install required Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    
    dependencies = [
        "torch",
        "transformers",
        "peft", 
        "ollama",
        "rouge-score",
        "numpy",
        "psutil"
    ]
    
    optional_deps = [
        "gputil"  # For GPU monitoring
    ]
    
    # Install required dependencies
    for dep in dependencies:
        try:
            print(f"Installing {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {dep}")
            return False
    
    # Install optional dependencies (don't fail if these don't work)
    for dep in optional_deps:
        try:
            print(f"Installing {dep} (optional)...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
        except subprocess.CalledProcessError:
            print(f"⚠️ Could not install {dep} (optional)")
    
    print("✓ Python dependencies installed")
    return True

def create_comparison_config() -> str:
    """Create a configuration file for the comparison"""
    config = {
        "test_config": {
            "phi2_finetuned_path": "c:/AI_Training/models/fine-tuned/phi2-optimized",
            "cline_optimal_model": "cline-optimal:latest",
            "warmup_iterations": 2,
            "test_iterations": 5,
            "max_new_tokens": 512,
            "temperature": 0.7,
            "results_dir": "c:/AI_Training/comparison_results"
        },
        "test_levels": [
            "basic",
            "intermediate", 
            "advanced",
            "autonomous"
        ],
        "created": time.time()
    }
    
    config_path = "c:/AI_Training/comparison_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Configuration saved to {config_path}")
    return config_path

def main():
    """Main setup function"""
    print("🚀 Cline-Optimal Model Comparison Setup")
    print("=" * 60)
    
    # Step 1: Check prerequisites
    print("\n📋 Checking prerequisites...")
    
    ollama_installed = check_ollama_installation()
    if not ollama_installed:
        print("\n❌ Ollama is required but not found.")
        print("Please install Ollama from: https://ollama.ai/")
        return False
    
    # Step 2: Check if model already exists
    model_exists = check_cline_optimal_model()
    
    # Step 3: Install Python dependencies
    if not install_python_dependencies():
        print("\n❌ Failed to install Python dependencies")
        return False
    
    # Step 4: Set up cline-optimal model if needed
    if not model_exists:
        print("\n🔧 Setting up cline-optimal model...")
        print("This will:")
        print("- Download qwen2.5-coder:7b-instruct-q4_K_M base model")
        print("- Create cline-optimal:latest with RTX 3090 optimizations")
        print("- Set up environment variables")
        print("- Test the model")
        
        proceed = input("\nProceed with setup? (y/N): ").strip().lower()
        if proceed != 'y':
            print("Setup cancelled")
            return False
        
        if not run_cline_optimal_setup():
            print("\n❌ Failed to set up cline-optimal model")
            return False
    else:
        print("\n✓ cline-optimal model already available")
    
    # Step 5: Test functionality
    test_results = test_model_functionality()
    
    # Step 6: Create configuration
    config_path = create_comparison_config()
    
    # Step 7: Summary and next steps
    print("\n" + "=" * 60)
    print("✅ SETUP COMPLETE!")
    print("=" * 60)
    
    print(f"\n📊 Model Availability:")
    print(f"  Phi-2 Fine-tuned: {'✓' if test_results['phi2_available'] else '✗'}")
    print(f"  Cline-Optimal: {'✓' if test_results['cline_optimal_available'] else '✗'}")
    
    if test_results['phi2_available'] and test_results['cline_optimal_available']:
        print(f"\n🎯 Ready for comparison testing!")
        print(f"\nNext steps:")
        print(f"1. Run the comparison:")
        print(f"   python comprehensive_model_comparison.py")
        print(f"\n2. Or run with custom settings:")
        print(f"   python -c \"from comprehensive_model_comparison import main; main()\"")
        
    elif test_results['cline_optimal_available']:
        print(f"\n⚠️ Only cline-optimal is available")
        print(f"The comparison will run in limited mode without Phi-2")
        
    else:
        print(f"\n❌ Setup issues detected")
        print(f"Please check the error messages above")
    
    print(f"\n📁 Configuration: {config_path}")
    print("=" * 60)
    
    return test_results['cline_optimal_available']

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
