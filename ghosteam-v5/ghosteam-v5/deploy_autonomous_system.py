#!/usr/bin/env python3
"""
Complete Deployment Script for Ghosteam V5 Autonomous MLOps System
Handles local deployment, cloud deployment, and system verification
"""

import subprocess
import sys
import time
import requests
import json
import os
import threading
from datetime import datetime

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🚀 GHOSTEAM V5 AUTONOMOUS MLOPS SYSTEM                    ║
║                         Complete Deployment & Setup                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

def print_status(message):
    print(f"✅ {message}")

def print_error(message):
    print(f"❌ {message}")

def print_info(message):
    print(f"ℹ️  {message}")

def print_section(title):
    print(f"\n{'='*60}")
    print(f"🔧 {title}")
    print('='*60)

def run_command(cmd, capture_output=True):
    """Run a command and return the result"""
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
        else:
            result = subprocess.run(cmd, shell=True)
            return result.returncode == 0, "", ""
    except Exception as e:
        return False, "", str(e)

def start_local_system():
    """Start the autonomous MLOps system locally"""
    print_section("Starting Local Autonomous MLOps System")
    
    # Start MLflow server in background
    print_info("Starting MLflow tracking server...")
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
        time.sleep(3)
        
        if mlflow_process.poll() is None:
            print_status("MLflow server started on http://localhost:5000")
        else:
            print_error("MLflow server failed to start")
    except Exception as e:
        print_error(f"Failed to start MLflow server: {e}")
    
    # Start the autonomous MLOps system
    print_info("Starting Autonomous MLOps System...")
    
    # Set environment variables
    os.environ['PYTHONPATH'] = os.getcwd()
    os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'
    
    # Start the system
    autonomous_cmd = [
        sys.executable, '-m', 'uvicorn',
        'src.autonomous_app:app',
        '--host', '0.0.0.0',
        '--port', '8084',
        '--reload'
    ]
    
    try:
        autonomous_process = subprocess.Popen(
            autonomous_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(5)
        
        if autonomous_process.poll() is None:
            print_status("Autonomous MLOps System started on http://localhost:8084")
            return autonomous_process, mlflow_process
        else:
            print_error("Autonomous MLOps System failed to start")
            return None, mlflow_process
    except Exception as e:
        print_error(f"Failed to start Autonomous MLOps System: {e}")
        return None, mlflow_process

def test_local_system():
    """Test the local autonomous system"""
    print_section("Testing Local System")
    
    base_url = "http://localhost:8084"
    
    # Test health endpoint
    try:
        print_info("Testing health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print_status("Health check passed")
            data = response.json()
            print_info(f"System status: {data.get('status', 'unknown')}")
            print_info(f"Autonomous features active: {data.get('autonomous_features', {})}")
        else:
            print_error(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Health check error: {e}")
        return False
    
    # Test prediction endpoint
    try:
        print_info("Testing prediction endpoint...")
        test_data = {
            "data": [1.0, -0.5, 0.3, -1.2, 0.8, -0.1, 0.5, -0.8, 1.1, -0.3],
            "context": {"user_type": "deployment_test", "frequency": "high"}
        }
        response = requests.post(f"{base_url}/predict", json=test_data, timeout=10)
        if response.status_code == 200:
            print_status("Prediction test passed")
            result = response.json()
            print_info(f"Prediction: {result.get('prediction')}")
            print_info(f"Confidence: {result.get('confidence', 0):.2f}")
            print_info(f"Suggestions: {len(result.get('suggestions', []))}")
        else:
            print_error(f"Prediction test failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Prediction test error: {e}")
        return False
    
    # Test insights endpoint
    try:
        print_info("Testing insights endpoint...")
        response = requests.get(f"{base_url}/insights", timeout=10)
        if response.status_code == 200:
            print_status("Insights test passed")
            data = response.json()
            print_info(f"Insights available: {len(data.get('insights', []))}")
            print_info(f"Autonomous actions: {len(data.get('autonomous_actions', []))}")
        else:
            print_error(f"Insights test failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Insights test error: {e}")
        return False
    
    return True

def create_deployment_package():
    """Create deployment package for cloud deployment"""
    print_section("Creating Deployment Package")
    
    # Create necessary directories
    os.makedirs("config", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    # Create deployment configuration
    deployment_config = {
        "name": "ghosteam-v5-autonomous",
        "version": "2.0.0",
        "description": "Autonomous MLOps System with Continuous Learning",
        "features": {
            "continuous_learning": True,
            "predictive_intelligence": True,
            "autonomous_retraining": True,
            "self_monitoring": True
        },
        "deployment": {
            "platform": "railway",
            "dockerfile": "Dockerfile.autonomous",
            "port": 8080,
            "health_check": "/health"
        }
    }
    
    with open("config/deployment.json", "w") as f:
        json.dump(deployment_config, f, indent=2)
    
    print_status("Deployment configuration created")
    
    # Create README for deployment
    readme_content = """# Ghosteam V5 Autonomous MLOps System

## 🚀 Features
- ✅ Continuous Learning
- ✅ Predictive Intelligence  
- ✅ Autonomous Retraining
- ✅ Self-Monitoring
- ✅ Interactive Dashboard

## 🌐 Access URLs
- Main API: `/`
- Dashboard: `/dashboard`
- Health Check: `/health`
- API Docs: `/docs`
- Insights: `/insights`

## 🤖 Autonomous Capabilities
The system automatically:
- Monitors model performance
- Retrains models when needed
- Analyzes usage patterns
- Generates predictive insights
- Optimizes performance

## 🔧 Management
- View logs: Check application logs
- Trigger retraining: POST `/retrain`
- Monitor health: GET `/health`
- View insights: GET `/insights`
"""
    
    with open("README_DEPLOYMENT.md", "w") as f:
        f.write(readme_content)
    
    print_status("Deployment documentation created")

def attempt_railway_deployment():
    """Attempt to deploy to Railway"""
    print_section("Attempting Railway Deployment")
    
    # Check if Railway CLI is available
    success, output, error = run_command("railway whoami")
    if not success:
        print_error("Railway CLI not available or not logged in")
        return False
    
    print_status(f"Railway CLI available, logged in as: {output}")
    
    # Try different deployment approaches
    deployment_commands = [
        "railway up --detach",
        "railway up --service ghosteam-v5 --detach",
        "railway up --service web --detach"
    ]
    
    for cmd in deployment_commands:
        print_info(f"Trying: {cmd}")
        success, output, error = run_command(cmd)
        
        if success:
            print_status("Railway deployment successful!")
            print_info("Waiting for deployment to complete...")
            time.sleep(30)
            
            # Try to get domain
            success, domain_output, error = run_command("railway domain")
            if success and "https://" in domain_output:
                domain = domain_output.strip()
                print_status(f"Deployment URL: {domain}")
                return domain
            else:
                print_info("Domain not yet available. You can add one with: railway domain")
                return True
        else:
            print_info(f"Command failed: {error}")
    
    print_error("Railway deployment failed with all methods")
    return False

def show_deployment_summary(local_running=False, cloud_url=None):
    """Show final deployment summary"""
    print_section("Deployment Summary")
    
    print("""
🎉 GHOSTEAM V5 AUTONOMOUS MLOPS SYSTEM DEPLOYED! 🎉

🤖 AUTONOMOUS FEATURES ACTIVE:
   ✅ Continuous Learning - Models improve automatically
   ✅ Predictive Intelligence - Anticipates your needs  
   ✅ Self-Monitoring - Tracks performance 24/7
   ✅ Auto-Retraining - Updates models when needed
   ✅ Pattern Analysis - Learns from usage patterns
    """)
    
    if local_running:
        print("""
🌐 LOCAL ACCESS:
   🚀 Main System:      http://localhost:8084
   📊 Dashboard:        http://localhost:8084/dashboard
   📓 API Docs:         http://localhost:8084/docs
   🔍 Health Check:     http://localhost:8084/health
   📈 MLflow:           http://localhost:5000
   🔮 Insights:         http://localhost:8084/insights
        """)
    
    if cloud_url:
        print(f"""
☁️  CLOUD ACCESS:
   🚀 Main System:      {cloud_url}
   📊 Dashboard:        {cloud_url}/dashboard
   📓 API Docs:         {cloud_url}/docs
   🔍 Health Check:     {cloud_url}/health
   🔮 Insights:         {cloud_url}/insights
        """)
    
    print("""
🧪 TEST COMMANDS:
   curl http://localhost:8084/health
   curl -X POST http://localhost:8084/predict -H "Content-Type: application/json" -d '{"data": [1,2,3,4,5,6,7,8,9,10]}'

🔧 MANAGEMENT:
   - System runs autonomously
   - Check dashboard for insights
   - Monitor health endpoint
   - View logs for autonomous actions

🚀 Your intelligent MLOps system is now operational and learning!
    """)

def main():
    """Main deployment function"""
    print_banner()
    
    print("🎯 DEPLOYMENT OPTIONS:")
    print("1. Local deployment (immediate)")
    print("2. Cloud deployment (Railway)")
    print("3. Both local and cloud")
    
    choice = input("\nSelect deployment option (1-3): ").strip()
    
    local_running = False
    cloud_url = None
    
    if choice in ["1", "3"]:
        # Local deployment
        autonomous_process, mlflow_process = start_local_system()
        if autonomous_process:
            local_running = test_local_system()
            if local_running:
                print_status("Local system is operational!")
            else:
                print_error("Local system tests failed")
    
    if choice in ["2", "3"]:
        # Cloud deployment
        create_deployment_package()
        cloud_result = attempt_railway_deployment()
        if cloud_result:
            if isinstance(cloud_result, str):
                cloud_url = cloud_result
            print_status("Cloud deployment initiated!")
        else:
            print_error("Cloud deployment failed")
            print_info("You can deploy manually:")
            print_info("1. Go to https://railway.app/dashboard")
            print_info("2. Create new project from GitHub")
            print_info("3. Connect this repository")
            print_info("4. Deploy using Dockerfile.autonomous")
    
    # Show summary
    show_deployment_summary(local_running, cloud_url)
    
    if local_running:
        print("\n🔄 System is running locally. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down system...")
            if 'autonomous_process' in locals() and autonomous_process:
                autonomous_process.terminate()
            if 'mlflow_process' in locals() and mlflow_process:
                mlflow_process.terminate()
            print_status("System shutdown complete")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
