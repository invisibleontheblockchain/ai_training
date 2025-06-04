"""
Quick Setup for AI Dashboard
Install only essential requirements for immediate testing
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a Python package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("🚀 Quick AI Dashboard Setup")
    print("Installing essential packages for immediate testing...")
    print("=" * 50)
    
    # Essential packages for basic functionality
    essential_packages = [
        "streamlit>=1.28.0",
        "plotly>=5.17.0", 
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "psutil>=5.9.0"
    ]
    
    # Try to install PyTorch with CUDA
    print("\n📦 Installing PyTorch with CUDA support...")
    torch_success = install_package("torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    if torch_success:
        print("✅ PyTorch with CUDA installed")
    else:
        print("⚠️  Installing CPU-only PyTorch...")
        install_package("torch torchvision torchaudio")
    
    # Install essential packages
    print("\n📦 Installing essential dashboard packages...")
    failed_packages = []
    
    for package in essential_packages:
        print(f"Installing {package.split('>=')[0]}...")
        if install_package(package):
            print(f"✅ {package.split('>=')[0]}")
        else:
            print(f"❌ {package.split('>=')[0]}")
            failed_packages.append(package)
    
    # Optional GPU monitoring
    print("\n📦 Installing optional GPU monitoring...")
    optional_packages = ["GPUtil", "transformers"]
    
    for package in optional_packages:
        if install_package(package):
            print(f"✅ {package}")
        else:
            print(f"⚠️  {package} (optional)")
    
    # Summary
    print("\n" + "=" * 50)
    if failed_packages:
        print(f"❌ Failed to install: {', '.join(failed_packages)}")
        print("You may need to install these manually.")
    else:
        print("✅ Essential packages installed successfully!")
    
    print("\n🎯 Next steps:")
    print("1. Run: python test_dashboard.py")
    print("2. If tests pass, run: .\\launch_ai_dashboard.ps1")
    print("3. Or directly: streamlit run ai_dashboard.py")

if __name__ == "__main__":
    main()
