"""
Quick setup script for enhanced benchmark system
"""

import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required packages"""
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.35.0", 
        "numpy",
        "psutil",
        "GPUtil",
    ]
    
    print("📦 Installing requirements...")
    for req in requirements:
        try:
            print(f"Installing {req}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", req, "--quiet"
            ])
            print(f"✅ {req}")
        except Exception as e:
            print(f"❌ Failed to install {req}: {e}")
            print(f"   You may need to install manually: pip install {req}")

def check_existing_files():
    """Check if required files exist"""
    required_files = [
        "models/system_prompt_injector.py",
        "models/enhanced_evaluation.py", 
        "models/enhanced_model_benchmark.py",
        "run_enhanced_benchmark.py"
    ]
    
    print("\n📁 Checking required files...")
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing!")
            all_exist = False
    
    return all_exist

def main():
    print("🛠️  Setting up Enhanced AI Benchmark System")
    print("=" * 50)
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}")
    
    # Check for existing files
    files_exist = check_existing_files()
    
    if not files_exist:
        print("\n⚠️  Some required files are missing!")
        print("Please ensure all files have been created properly.")
        return
    
    # Install requirements
    install_requirements()
    
    print("\n🚀 Setup complete!")
    print("\nTo run the enhanced benchmark:")
    print("   python run_enhanced_benchmark.py")
    print("\nTo test individual components:")
    print("   python models/system_prompt_injector.py")
    print("   python models/enhanced_model_benchmark.py")

if __name__ == "__main__":
    main()
