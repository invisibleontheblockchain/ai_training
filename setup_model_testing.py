#!/usr/bin/env python3
"""
Quick Setup and Validation Script
=================================

This script validates that your testing environment is properly configured
and all dependencies are available.

Usage:
    python setup_model_testing.py [--install-deps]

Created: June 4, 2025
"""

import sys
import os
import subprocess
import importlib
from pathlib import Path

def check_python_version():
    """Check if Python version is adequate"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ required.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
    return True

def check_dependencies():
    """Check if required dependencies are available"""
    print("\n📦 Checking dependencies...")
    
    dependencies = {
        "required": [
            ("json", "json"),
            ("time", "time"), 
            ("pathlib", "pathlib"),
            ("logging", "logging"),
            ("argparse", "argparse"),
            ("datetime", "datetime"),
            ("typing", "typing")
        ],
        "model_specific": [
            ("transformers", "transformers"),
            ("torch", "torch"),
            ("peft", "peft"),
            ("ollama", "ollama"),
            ("numpy", "numpy")
        ],
        "optional": [
            ("GPUtil", "GPUtil"),
            ("psutil", "psutil"),
            ("rouge_score", "rouge-score"),
            ("matplotlib", "matplotlib"),
            ("pandas", "pandas")
        ]
    }
    
    results = {"required": [], "model_specific": [], "optional": []}
    
    for category, deps in dependencies.items():
        print(f"\n{category.title()} dependencies:")
        for module_name, package_name in deps:
            try:
                importlib.import_module(module_name)
                print(f"  ✅ {package_name}")
                results[category].append((package_name, True))
            except ImportError:
                print(f"  ❌ {package_name}")
                results[category].append((package_name, False))
    
    return results

def check_file_structure():
    """Check if required files exist"""
    print("\n📁 Checking file structure...")
    
    required_files = [
        "comprehensive_model_test_prompt.py",
        "test_your_model.py",
        "MODEL_TESTING_GUIDE.md"
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file}")
            missing_files.append(file)
    
    return missing_files

def check_model_paths():
    """Check if model paths exist"""
    print("\n🤖 Checking model paths...")
    
    paths_to_check = [
        ("Phi-2 base models", "c:/AI_Training/models"),
        ("Fine-tuned models", "c:/AI_Training/models/fine-tuned"),
        ("Results directory", "c:/AI_Training/test_results")
    ]
    
    for description, path in paths_to_check:
        if os.path.exists(path):
            print(f"  ✅ {description}: {path}")
            # List contents if it's a models directory
            if "models" in path:
                try:
                    contents = os.listdir(path)
                    if contents:
                        print(f"      Contents: {', '.join(contents[:5])}")
                        if len(contents) > 5:
                            print(f"      ... and {len(contents) - 5} more")
                    else:
                        print("      (empty)")
                except PermissionError:
                    print("      (permission denied)")
        else:
            print(f"  ❌ {description}: {path}")

def check_ollama():
    """Check if Ollama is available and has cline-optimal"""
    print("\n🦙 Checking Ollama...")
    
    try:
        # Check if ollama command is available
        result = subprocess.run(["ollama", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"  ✅ Ollama version: {result.stdout.strip()}")
            
            # Check for cline-optimal model
            try:
                import ollama
                models = ollama.list()
                model_names = [m['name'] for m in models.get('models', [])]
                
                if 'cline-optimal:latest' in model_names:
                    print("  ✅ cline-optimal:latest model found")
                else:
                    print("  ❌ cline-optimal:latest model not found")
                    print("      Available models:")
                    for model in model_names[:5]:
                        print(f"        - {model}")
                    if len(model_names) > 5:
                        print(f"        ... and {len(model_names) - 5} more")
                        
            except Exception as e:
                print(f"  ⚠️  Could not list Ollama models: {e}")
                
        else:
            print(f"  ❌ Ollama not available: {result.stderr}")
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("  ❌ Ollama command not found or not responding")

def create_test_results_directory():
    """Create test results directory if it doesn't exist"""
    results_dir = Path("test_results")
    if not results_dir.exists():
        results_dir.mkdir(parents=True)
        print(f"📁 Created directory: {results_dir}")
    return results_dir

def install_dependencies(deps_to_install):
    """Install missing dependencies"""
    if not deps_to_install:
        print("No dependencies to install.")
        return
        
    print(f"\n📦 Installing {len(deps_to_install)} packages...")
    
    for package in deps_to_install:
        print(f"Installing {package}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True)
            print(f"  ✅ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to install {package}: {e}")

def generate_quick_test():
    """Generate a quick test command"""
    print("\n🧪 Quick Test Commands:")
    print("=" * 40)
    
    commands = [
        ("Test Phi-2 with basic prompts", "python test_your_model.py --model phi2 --level basic"),
        ("Test cline-optimal with basic prompts", "python test_your_model.py --model cline-optimal --level basic"),
        ("Compare both models", "python test_your_model.py --model both --level intermediate"),
        ("Test autonomous capabilities", "python test_your_model.py --model both --level autonomous"),
        ("Full comprehensive test", "python test_your_model.py --model both --level all")
    ]
    
    for description, command in commands:
        print(f"{description}:")
        print(f"  {command}")
        print()

def main():
    """Main setup function"""
    print("🚀 Model Testing Environment Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Check dependencies
    dep_results = check_dependencies()
    
    # Check file structure
    missing_files = check_file_structure()
    
    # Check model paths
    check_model_paths()
    
    # Check Ollama
    check_ollama()
    
    # Create results directory
    create_test_results_directory()
    
    # Summary
    print("\n📊 SETUP SUMMARY")
    print("=" * 30)
    
    # Count missing dependencies
    missing_required = [pkg for pkg, available in dep_results["required"] if not available]
    missing_model = [pkg for pkg, available in dep_results["model_specific"] if not available]
    missing_optional = [pkg for pkg, available in dep_results["optional"] if not available]
    
    if missing_required:
        print(f"❌ Missing REQUIRED packages: {', '.join(missing_required)}")
        return 1
    else:
        print("✅ All required packages available")
    
    if missing_model:
        print(f"⚠️  Missing model packages: {', '.join(missing_model)}")
        print("   These are needed for specific model testing")
        
        if "--install-deps" in sys.argv:
            install_dependencies(missing_model)
    
    if missing_optional:
        print(f"ℹ️  Missing optional packages: {', '.join(missing_optional)}")
        print("   These provide enhanced features but aren't required")
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        print("   Please ensure all required files are in the current directory")
        return 1
    
    print("\n✅ Environment setup complete!")
    
    # Generate test commands
    generate_quick_test()
    
    # Final recommendations
    print("🎯 RECOMMENDATIONS:")
    print("-" * 20)
    
    if missing_model:
        print("1. Install missing model packages for full functionality")
    if missing_optional:
        print("2. Consider installing optional packages for enhanced features")
    
    print("3. Start with basic tests to verify everything works")
    print("4. Review MODEL_TESTING_GUIDE.md for detailed usage instructions")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
