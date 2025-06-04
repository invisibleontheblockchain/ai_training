"""
AI Dashboard Test Script
Test basic functionality before launching the full dashboard
"""

import sys
import subprocess
import importlib
from pathlib import Path

def test_imports():
    """Test if required modules can be imported"""
    print("🧪 Testing Python imports...")
    
    required_modules = [
        "streamlit",
        "plotly",
        "pandas", 
        "numpy",
        "torch",
        "psutil"
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module}")
            failed_imports.append(module)
    
    return failed_imports

def test_torch_cuda():
    """Test PyTorch CUDA availability"""
    print("\n🔥 Testing PyTorch CUDA...")
    
    try:
        import torch
        print(f"  PyTorch Version: {torch.__version__}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("  ⚠️  CUDA not available")
            return False
    except ImportError:
        print("  ❌ PyTorch not installed")
        return False

def test_gpu_monitoring():
    """Test GPU monitoring capabilities"""
    print("\n📊 Testing GPU monitoring...")
    
    try:
        import psutil
        print(f"  CPU Usage: {psutil.cpu_percent()}%")
        print(f"  Memory Usage: {psutil.virtual_memory().percent}%")
        print("  ✅ Basic system monitoring working")
        
        # Test advanced GPU monitoring
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                print(f"  GPU Load: {gpu.load * 100:.1f}%")
                print(f"  GPU Memory: {gpu.memoryUsed}/{gpu.memoryTotal} MB")
                print(f"  GPU Temperature: {gpu.temperature}°C")
                print("  ✅ Advanced GPU monitoring working")
            else:
                print("  ⚠️  No GPUs detected by GPUtil")
        except ImportError:
            print("  ⚠️  GPUtil not available (optional)")
            
        return True
    except Exception as e:
        print(f"  ❌ Monitoring test failed: {e}")
        return False

def test_model_modules():
    """Test if custom model modules are available"""
    print("\n🤖 Testing custom model modules...")
    
    models_dir = Path("models")
    if not models_dir.exists():
        print("  ⚠️  models/ directory not found")
        return False
    
    required_files = [
        "enhanced_model_benchmark.py",
        "system_prompt_injector.py", 
        "enhanced_evaluation.py"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = models_dir / file
        if file_path.exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file}")
            missing_files.append(file)
    
    return len(missing_files) == 0

def test_streamlit():
    """Test if Streamlit can start"""
    print("\n🌐 Testing Streamlit...")
    
    try:
        result = subprocess.run(
            ["streamlit", "--version"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        if result.returncode == 0:
            print(f"  ✅ Streamlit version: {result.stdout.strip()}")
            return True
        else:
            print(f"  ❌ Streamlit test failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("  ⚠️  Streamlit test timed out")
        return False
    except FileNotFoundError:
        print("  ❌ Streamlit not found in PATH")
        return False

def main():
    """Run all tests"""
    print("🚀 AI Dashboard Test Suite")
    print("=" * 50)
    
    # Run tests
    failed_imports = test_imports()
    cuda_available = test_torch_cuda()
    monitoring_ok = test_gpu_monitoring()
    models_ok = test_model_modules()
    streamlit_ok = test_streamlit()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    
    if failed_imports:
        print(f"❌ Missing modules: {', '.join(failed_imports)}")
        print("   Run: pip install -r dashboard_requirements.txt")
    else:
        print("✅ All required modules available")
    
    if cuda_available:
        print("✅ CUDA and RTX 3090 ready")
    else:
        print("⚠️  CUDA not available - GPU features limited")
    
    if monitoring_ok:
        print("✅ System monitoring ready")
    else:
        print("❌ System monitoring issues")
    
    if models_ok:
        print("✅ Custom model modules ready")
    else:
        print("⚠️  Some model modules missing - limited functionality")
    
    if streamlit_ok:
        print("✅ Streamlit ready to launch")
    else:
        print("❌ Streamlit issues - check installation")
    
    # Overall status
    if not failed_imports and streamlit_ok:
        print("\n🎉 Dashboard ready to launch!")
        print("   Run: .\\launch_ai_dashboard.ps1")
    else:
        print("\n🔧 Setup required before launching dashboard")
        print("   Run: .\\launch_ai_dashboard.ps1 -Install")

if __name__ == "__main__":
    main()
