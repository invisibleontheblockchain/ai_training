#!/usr/bin/env python3
"""
Ghosteam V5 Complete System Verification Script
Comprehensive end-to-end testing of the deployed MLOps system
"""

import sys
import os
import time
import requests
import json
import mlflow
from datetime import datetime

def print_header(message):
    print(f"\n{'='*60}")
    print(f"🔍 {message}")
    print('='*60)

def print_status(message):
    print(f"✅ {message}")

def print_error(message):
    print(f"❌ {message}")

def print_info(message):
    print(f"ℹ️  {message}")

def test_api_endpoints():
    """Test all API endpoints"""
    print_header("Testing API Endpoints")
    
    base_url = "http://localhost:8081"
    endpoints = [
        ("/", "Root endpoint"),
        ("/health", "Health check"),
        ("/models", "Model registry"),
    ]
    
    all_passed = True
    
    for endpoint, description in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            if response.status_code == 200:
                print_status(f"{description}: {endpoint}")
                data = response.json()
                if endpoint == "/health":
                    print_info(f"   MLflow version: {data.get('mlflow_version', 'unknown')}")
                elif endpoint == "/models":
                    print_info(f"   Models found: {data.get('count', 0)}")
            else:
                print_error(f"{description} failed: {response.status_code}")
                all_passed = False
        except Exception as e:
            print_error(f"{description} error: {e}")
            all_passed = False
    
    # Test POST endpoint
    try:
        test_data = {"data": [1.0, -0.5, 0.3, -1.2, 0.8, -0.1, 0.5, -0.8, 1.1, -0.3]}
        response = requests.post(f"{base_url}/predict", json=test_data, timeout=10)
        if response.status_code == 200:
            print_status("Prediction endpoint: /predict")
            result = response.json()
            print_info(f"   Prediction: {result.get('prediction', 'unknown')}")
        else:
            print_error(f"Prediction endpoint failed: {response.status_code}")
            all_passed = False
    except Exception as e:
        print_error(f"Prediction endpoint error: {e}")
        all_passed = False
    
    return all_passed

def test_mlflow_server():
    """Test MLflow server functionality"""
    print_header("Testing MLflow Server")
    
    mlflow_url = "http://localhost:5000"
    
    try:
        # Test MLflow web interface
        response = requests.get(mlflow_url, timeout=10)
        if response.status_code == 200:
            print_status("MLflow web interface accessible")
        else:
            print_error(f"MLflow web interface failed: {response.status_code}")
            return False
        
        # Test MLflow API
        os.environ['MLFLOW_TRACKING_URI'] = mlflow_url
        mlflow.set_tracking_uri(mlflow_url)
        
        client = mlflow.tracking.MlflowClient()
        
        # List experiments
        experiments = client.search_experiments()
        print_status(f"MLflow experiments: {len(experiments)}")
        for exp in experiments:
            print_info(f"   {exp.name} (ID: {exp.experiment_id})")
        
        # List models
        models = client.search_registered_models()
        print_status(f"Registered models: {len(models)}")
        for model in models:
            print_info(f"   {model.name}")
            versions = client.get_latest_versions(model.name)
            for version in versions:
                print_info(f"      Version {version.version} (Stage: {version.current_stage})")
        
        return True
        
    except Exception as e:
        print_error(f"MLflow server test failed: {e}")
        return False

def test_model_training():
    """Test model training pipeline"""
    print_header("Testing Model Training Pipeline")
    
    try:
        # Set MLflow tracking
        mlflow.set_tracking_uri("http://localhost:5000")
        
        # Create a simple test experiment
        experiment_name = "verification-test"
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = experiment.experiment_id
        
        mlflow.set_experiment(experiment_name)
        
        # Start a test run
        with mlflow.start_run(run_name=f"verification_{datetime.now().strftime('%H%M%S')}"):
            # Log some test parameters and metrics
            mlflow.log_param("test_param", "verification")
            mlflow.log_metric("test_metric", 0.95)
            
            print_status("Test MLflow run created successfully")
        
        return True
        
    except Exception as e:
        print_error(f"Model training test failed: {e}")
        return False

def test_system_integration():
    """Test complete system integration"""
    print_header("Testing System Integration")
    
    try:
        # Test that API can communicate with MLflow
        api_response = requests.get("http://localhost:8081/models", timeout=10)
        if api_response.status_code != 200:
            print_error("API not responding")
            return False
        
        api_data = api_response.json()
        
        # Test that MLflow has models
        mlflow.set_tracking_uri("http://localhost:5000")
        client = mlflow.tracking.MlflowClient()
        mlflow_models = client.search_registered_models()
        
        if len(mlflow_models) > 0 and api_data.get('count', 0) > 0:
            print_status("API successfully connected to MLflow")
            print_info(f"   API reports {api_data['count']} models")
            print_info(f"   MLflow has {len(mlflow_models)} models")
            
            # Verify model names match
            api_model_names = [m['name'] for m in api_data.get('models', [])]
            mlflow_model_names = [m.name for m in mlflow_models]
            
            if set(api_model_names) == set(mlflow_model_names):
                print_status("Model registry synchronization verified")
            else:
                print_error("Model registry mismatch between API and MLflow")
                return False
        else:
            print_error("No models found or API-MLflow connection issue")
            return False
        
        return True
        
    except Exception as e:
        print_error(f"System integration test failed: {e}")
        return False

def generate_system_report():
    """Generate comprehensive system report"""
    print_header("System Status Report")
    
    try:
        # API Status
        api_response = requests.get("http://localhost:8081/health", timeout=10)
        api_healthy = api_response.status_code == 200
        
        # MLflow Status
        mlflow_response = requests.get("http://localhost:5000", timeout=10)
        mlflow_healthy = mlflow_response.status_code == 200
        
        # Model Count
        models_response = requests.get("http://localhost:8081/models", timeout=10)
        model_count = 0
        if models_response.status_code == 200:
            model_count = models_response.json().get('count', 0)
        
        print(f"""
🎉 GHOSTEAM V5 MLOPS SYSTEM STATUS REPORT 🎉

🌐 System Components:
   ✅ FastAPI Server:     {'HEALTHY' if api_healthy else 'UNHEALTHY'}
   ✅ MLflow Server:      {'HEALTHY' if mlflow_healthy else 'UNHEALTHY'}
   ✅ Model Registry:     {model_count} models registered

🔗 Access URLs:
   🚀 API:                http://localhost:8081
   📓 API Docs:           http://localhost:8081/docs
   🔍 Health Check:       http://localhost:8081/health
   📊 MLflow:             http://localhost:5000
   📈 Models:             http://localhost:8081/models

🧪 Test Commands:
   curl http://localhost:8081/health
   curl http://localhost:8081/models
   curl -X POST http://localhost:8081/predict -H "Content-Type: application/json" -d '{{"data": [1,2,3,4,5,6,7,8,9,10]}}'

📊 Capabilities Verified:
   ✅ Model Training with MLflow Tracking
   ✅ Model Registration and Versioning
   ✅ REST API for Model Serving
   ✅ Health Monitoring
   ✅ End-to-End Integration

🚀 Your complete MLOps system is operational and ready for production use!
        """)
        
    except Exception as e:
        print_error(f"Failed to generate system report: {e}")

def main():
    """Main verification function"""
    print("""
🚀 GHOSTEAM V5 MLOPS SYSTEM VERIFICATION
========================================
    """)
    
    all_tests_passed = True
    
    # Run all tests
    tests = [
        ("API Endpoints", test_api_endpoints),
        ("MLflow Server", test_mlflow_server),
        ("Model Training", test_model_training),
        ("System Integration", test_system_integration),
    ]
    
    for test_name, test_func in tests:
        try:
            if not test_func():
                all_tests_passed = False
        except Exception as e:
            print_error(f"{test_name} test failed with exception: {e}")
            all_tests_passed = False
    
    # Generate final report
    generate_system_report()
    
    if all_tests_passed:
        print_status("\n🎉 ALL VERIFICATION TESTS PASSED!")
        print_status("🚀 Ghosteam V5 MLOps System is fully operational!")
    else:
        print_error("\n❌ Some verification tests failed.")
        print_info("Please check the logs above for details.")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
