"""
Quick Dependencies Installer
============================
Installs all required dependencies for the model comparison tool.

Run this first if you encounter import errors.
"""

import subprocess
import sys

def install_package(package_name, optional=False):
    """Install a Python package using pip"""
    try:
        print(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        if optional:
            print(f"⚠️ {package_name} (optional) failed to install: {e}")
            return False
        else:
            print(f"❌ {package_name} (required) failed to install: {e}")
            return False

def main():
    """Install all dependencies"""
    print("📦 Installing Model Comparison Dependencies")
    print("=" * 50)
    
    # Required packages
    required_packages = [
        "torch",
        "transformers>=4.30.0",
        "peft",
        "ollama",
        "numpy",
        "psutil"
    ]
    
    # Optional packages (for enhanced functionality)
    optional_packages = [
        "rouge-score",  # For quality scoring
        "gputil",       # For GPU monitoring
        "accelerate",   # For model loading optimization
        "bitsandbytes", # For quantization
    ]
    
    failed_required = []
    failed_optional = []
    
    # Install required packages
    print("\n🔧 Installing required packages...")
    for package in required_packages:
        if not install_package(package):
            failed_required.append(package)
    
    # Install optional packages
    print("\n🎁 Installing optional packages...")
    for package in optional_packages:
        if not install_package(package, optional=True):
            failed_optional.append(package)
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Installation Summary")
    print("=" * 50)
    
    if not failed_required:
        print("✅ All required packages installed successfully!")
    else:
        print("❌ Some required packages failed to install:")
        for package in failed_required:
            print(f"  - {package}")
    
    if failed_optional:
        print("\n⚠️ Some optional packages failed to install:")
        for package in failed_optional:
            print(f"  - {package}")
        print("The comparison tool will work but with reduced functionality.")
    
    if not failed_required:
        print(f"\n🚀 Ready to run model comparison!")
        print(f"Next steps:")
        print(f"1. Set up models: python setup_comparison.py")
        print(f"2. Run comparison: python run_comparison.py")
    else:
        print(f"\n❌ Please resolve the required package installation issues first.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
