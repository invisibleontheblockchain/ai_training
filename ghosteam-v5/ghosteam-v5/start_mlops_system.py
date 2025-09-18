#!/usr/bin/env python3
"""
Ghosteam V5 MLOps System Startup Script
Starts the complete MLOps system locally or in cloud environment
"""

import sys
import os
import time
import subprocess
import threading
import signal
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def print_banner():
    print("""
   ______ __  __ ____   _____ _______ ______          __  __ __      __ _____ 
  / ____// / / // __ \ / ___//_  __/ / ____/         / / / /\ \    / // ____/
 / / __ / /_/ // / / / \__ \  / /   / __/    ______ / / / /  \ \  / //___ \  
/ /_/ // __  // /_/ / ___/ / / /   / /___   /_____// /_/ /    \ \/ / ____/ /  
\____//_/ /_/ \____/ /____/ /_/   /_____/          \____/      \__/ /_____/   

🚀 GHOSTEAM V5 MLOPS SYSTEM - COMPLETE DEPLOYMENT
""")

def print_status(message):
    print(f"✅ {message}")

def print_error(message):
    print(f"❌ {message}")

def print_info(message):
    print(f"ℹ️  {message}")

def check_dependencies():
    """Check if all required dependencies are available"""
    print("\n🔍 Checking Dependencies...")
    
    required_modules = [
        ('mlflow', 'MLflow'),
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
        ('redis', 'Redis'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy')
    ]
    
    all_available = True
    for module, name in required_modules:
        try:
            __import__(module)
            print_status(f"{name} available")
        except ImportError:
            print_error(f"{name} not available")
            all_available = False
    
    return all_available

def start_mlflow_server():
    """Start MLflow tracking server"""
    print("\n📊 Starting MLflow Tracking Server...")
    
    # Create mlruns directory if it doesn't exist
    os.makedirs('mlruns', exist_ok=True)
    
    # Start MLflow server in background
    mlflow_cmd = [
        sys.executable, '-m', 'mlflow', 'server',
        '--backend-store-uri', 'sqlite:///mlruns.db',
        '--default-artifact-root', './mlruns',
        '--host', '0.0.0.0',
        '--port', '5000'
    ]
    
    try:
        mlflow_process = subprocess.Popen(
            mlflow_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(3)  # Give it time to start
        
        if mlflow_process.poll() is None:
            print_status("MLflow server started on http://localhost:5000")
            return mlflow_process
        else:
            print_error("MLflow server failed to start")
            return None
    except Exception as e:
        print_error(f"Failed to start MLflow server: {e}")
        return None

def start_api_server():
    """Start the FastAPI server"""
    print("\n🚀 Starting FastAPI Server...")
    
    # Set environment variables
    os.environ['PYTHONPATH'] = os.getcwd()
    os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'
    
    # Start API server
    api_cmd = [
        sys.executable, '-m', 'uvicorn',
        'src.minimal_app:app',
        '--host', '0.0.0.0',
        '--port', '8080',
        '--reload'
    ]
    
    try:
        api_process = subprocess.Popen(
            api_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(3)  # Give it time to start
        
        if api_process.poll() is None:
            print_status("FastAPI server started on http://localhost:8080")
            return api_process
        else:
            print_error("FastAPI server failed to start")
            return None
    except Exception as e:
        print_error(f"Failed to start FastAPI server: {e}")
        return None

def test_system():
    """Test the running system"""
    print("\n🧪 Testing System...")
    
    try:
        import requests
        
        # Test API health
        response = requests.get('http://localhost:8080/health', timeout=10)
        if response.status_code == 200:
            print_status("API health check passed")
            data = response.json()
            print_info(f"MLflow version: {data.get('mlflow_version', 'unknown')}")
        else:
            print_error(f"API health check failed: {response.status_code}")
        
        # Test MLflow server
        response = requests.get('http://localhost:5000/health', timeout=10)
        if response.status_code == 200:
            print_status("MLflow server health check passed")
        else:
            print_info("MLflow server health endpoint not available (normal)")
        
        return True
        
    except Exception as e:
        print_error(f"System test failed: {e}")
        return False

def show_system_info():
    """Show system information and access URLs"""
    print(f"""
🎉 GHOSTEAM V5 MLOPS SYSTEM IS RUNNING! 🎉

🌐 Access Your System:
   🚀 API:              http://localhost:8080
   📓 API Docs:         http://localhost:8080/docs
   🔍 Health Check:     http://localhost:8080/health
   📊 MLflow:           http://localhost:5000
   📈 Models:           http://localhost:8080/models

🧪 Test Commands:
   curl http://localhost:8080/health
   curl http://localhost:8080/models
   curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d '{{"data": "test"}}'

📊 What's Running:
   ✅ FastAPI Model Serving API (with MLflow integration)
   ✅ MLflow Tracking Server (experiment tracking & model registry)
   ✅ Health monitoring endpoints
   ✅ Model prediction endpoints

🔧 System Management:
   Stop: Press Ctrl+C
   Logs: Check terminal output
   
🚀 Your MLOps system is now operational!

Press Ctrl+C to stop all services...
""")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\n\n🛑 Shutting down Ghosteam V5 MLOps System...")
    sys.exit(0)

def main():
    """Main function to start the complete MLOps system"""
    print_banner()
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Check dependencies
    if not check_dependencies():
        print_error("Missing dependencies. Please install required packages.")
        return False
    
    # Start MLflow server
    mlflow_process = start_mlflow_server()
    if not mlflow_process:
        return False
    
    # Start API server
    api_process = start_api_server()
    if not api_process:
        if mlflow_process:
            mlflow_process.terminate()
        return False
    
    # Wait a moment for services to fully start
    time.sleep(5)
    
    # Test system
    test_system()
    
    # Show system info
    show_system_info()
    
    try:
        # Keep the system running
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if mlflow_process.poll() is not None:
                print_error("MLflow server stopped unexpectedly")
                break
            
            if api_process.poll() is not None:
                print_error("API server stopped unexpectedly")
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        # Clean shutdown
        print("\n🛑 Stopping services...")
        if api_process and api_process.poll() is None:
            api_process.terminate()
            print_status("API server stopped")
        
        if mlflow_process and mlflow_process.poll() is None:
            mlflow_process.terminate()
            print_status("MLflow server stopped")
        
        print_status("Ghosteam V5 MLOps System shutdown complete")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
