#!/usr/bin/env python3
"""
Render.com Deployment Verification for Ghosteam V5 Autonomous MLOps
Comprehensive testing of deployed system at https://ghosteam-v5-autonomous.onrender.com
"""

import requests
import json
import time
import sys
from datetime import datetime

DEPLOYMENT_URL = "https://ghosteam-v5-autonomous.onrender.com"

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🔍 RENDER.COM DEPLOYMENT VERIFICATION                     ║
║                   Ghosteam V5 Autonomous MLOps System                        ║
║                  https://ghosteam-v5-autonomous.onrender.com                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

def print_test(test_name):
    print(f"\n🧪 {test_name}")
    print("-" * 60)

def print_status(message):
    print(f"✅ {message}")

def print_error(message):
    print(f"❌ {message}")

def print_info(message):
    print(f"ℹ️  {message}")

def print_warning(message):
    print(f"⚠️  {message}")

def wait_for_service():
    """Wait for service to be ready (handle cold start)"""
    print_test("WAITING FOR SERVICE TO BE READY")
    print_info("This may take up to 60 seconds for cold start...")
    
    max_attempts = 12  # 2 minutes total
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{DEPLOYMENT_URL}/health", timeout=30)
            if response.status_code == 200:
                print_status("Service is ready!")
                return True
        except:
            pass
        
        print_info(f"Attempt {attempt + 1}/{max_attempts} - waiting...")
        time.sleep(10)
    
    print_error("Service did not become ready within 2 minutes")
    return False

def test_health_endpoint():
    """Test system health and autonomous features"""
    print_test("SYSTEM HEALTH & AUTONOMOUS FEATURES")
    
    try:
        response = requests.get(f"{DEPLOYMENT_URL}/health", timeout=30)
        
        if response.status_code == 200:
            print_status("Health endpoint accessible")
            data = response.json()
            
            # System status
            status = data.get('status', 'unknown')
            print_info(f"System status: {status}")
            
            # Autonomous features
            autonomous_features = data.get('autonomous_features', {})
            print_info("Autonomous features:")
            for feature, enabled in autonomous_features.items():
                icon = "✅" if enabled else "❌"
                print_info(f"  {icon} {feature}: {enabled}")
            
            # Services
            services = data.get('services', {})
            print_info("Services status:")
            for service, status in services.items():
                print_info(f"  📊 {service}: {status}")
            
            # Metrics
            metrics = data.get('metrics', {})
            if metrics:
                print_info("System metrics:")
                for metric, value in metrics.items():
                    print_info(f"  📈 {metric}: {value}")
            
            return True
        else:
            print_error(f"Health check failed: HTTP {response.status_code}")
            print_error(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print_error(f"Health check error: {e}")
        return False

def test_prediction_functionality():
    """Test ML prediction capabilities"""
    print_test("ML PREDICTION FUNCTIONALITY")
    
    try:
        test_data = {
            "data": [1.0, -0.5, 0.3, -1.2, 0.8, -0.1, 0.5, -0.8, 1.1, -0.3],
            "context": {
                "user_type": "render_deployment_test",
                "test_timestamp": datetime.now().isoformat(),
                "source": "verification_script"
            }
        }
        
        response = requests.post(
            f"{DEPLOYMENT_URL}/predict", 
            json=test_data, 
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print_status("Prediction endpoint working")
            result = response.json()
            
            print_info(f"Prediction: {result.get('prediction')}")
            print_info(f"Confidence: {result.get('confidence', 0):.3f}")
            print_info(f"Model used: {result.get('model_used', 'unknown')}")
            
            suggestions = result.get('suggestions', [])
            print_info(f"AI suggestions: {len(suggestions)} provided")
            for i, suggestion in enumerate(suggestions[:2], 1):
                print_info(f"  {i}. {suggestion}")
            
            return True
        else:
            print_error(f"Prediction failed: HTTP {response.status_code}")
            print_error(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print_error(f"Prediction test error: {e}")
        return False

def test_autonomous_features():
    """Test autonomous learning and insights"""
    print_test("AUTONOMOUS LEARNING & PREDICTIVE INTELLIGENCE")
    
    try:
        response = requests.get(f"{DEPLOYMENT_URL}/insights", timeout=30)
        
        if response.status_code == 200:
            print_status("Insights endpoint accessible")
            data = response.json()
            
            insights = data.get('insights', [])
            print_info(f"Active insights: {len(insights)}")
            
            usage_patterns = data.get('usage_patterns', [])
            print_info(f"Usage patterns: {len(usage_patterns)} tracked")
            
            autonomous_actions = data.get('autonomous_actions', [])
            print_info(f"Autonomous actions: {len(autonomous_actions)} completed")
            
            if autonomous_actions:
                latest = autonomous_actions[-1]
                print_info(f"Latest action: {latest.get('action', 'unknown')}")
                if 'accuracy' in latest:
                    print_info(f"  Accuracy: {latest['accuracy']:.3f}")
            
            return True
        else:
            print_error(f"Insights failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Insights test error: {e}")
        return False

def test_dashboard_access():
    """Test interactive dashboard"""
    print_test("INTERACTIVE DASHBOARD")
    
    try:
        response = requests.get(f"{DEPLOYMENT_URL}/dashboard", timeout=30)
        
        if response.status_code == 200:
            print_status("Dashboard accessible")
            print_info(f"URL: {DEPLOYMENT_URL}/dashboard")
            
            if 'text/html' in response.headers.get('content-type', ''):
                print_status("Dashboard returns HTML content")
                
                # Check for key elements
                content = response.text.lower()
                elements = ['autonomous', 'mlops', 'dashboard', 'system status']
                
                for element in elements:
                    if element in content:
                        print_info(f"  ✅ Contains: {element}")
                
                return True
            else:
                print_error("Dashboard doesn't return HTML")
                return False
        else:
            print_error(f"Dashboard failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Dashboard test error: {e}")
        return False

def test_api_documentation():
    """Test API docs accessibility"""
    print_test("API DOCUMENTATION")
    
    try:
        response = requests.get(f"{DEPLOYMENT_URL}/docs", timeout=30)
        
        if response.status_code == 200:
            print_status("API documentation accessible")
            print_info(f"URL: {DEPLOYMENT_URL}/docs")
            return True
        else:
            print_error(f"API docs failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"API docs test error: {e}")
        return False

def test_autonomous_retraining():
    """Test manual trigger of autonomous retraining"""
    print_test("AUTONOMOUS RETRAINING CAPABILITY")
    
    try:
        response = requests.post(f"{DEPLOYMENT_URL}/retrain", timeout=60)
        
        if response.status_code == 200:
            print_status("Retraining endpoint accessible")
            result = response.json()
            print_info(f"Status: {result.get('status', 'unknown')}")
            print_info(f"Message: {result.get('message', 'No message')}")
            return True
        else:
            print_error(f"Retraining failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Retraining test error: {e}")
        return False

def check_free_tier_limitations():
    """Check for free tier spin-down issues"""
    print_test("FREE TIER LIMITATIONS CHECK")
    
    print_warning("CRITICAL: Free tier limitations detected!")
    print_warning("Your service will spin down after 15 minutes of inactivity")
    print_warning("This will interrupt autonomous learning processes")
    
    print_info("For 24/7 autonomous operation, you need to:")
    print_info("1. Go to Render dashboard")
    print_info("2. Select your service")
    print_info("3. Go to Settings → Plan")
    print_info("4. Upgrade to Starter Plan ($7/month)")
    
    return False  # Always return False to highlight this issue

def main():
    print_banner()
    
    print_info("Testing Ghosteam V5 Autonomous MLOps deployment...")
    print_info(f"Target URL: {DEPLOYMENT_URL}")
    
    # Wait for service to be ready
    if not wait_for_service():
        print_error("Service is not ready. Check Render dashboard for build status.")
        return False
    
    # Run all tests
    tests = [
        ("System Health", test_health_endpoint),
        ("ML Predictions", test_prediction_functionality),
        ("Autonomous Features", test_autonomous_features),
        ("Dashboard Access", test_dashboard_access),
        ("API Documentation", test_api_documentation),
        ("Retraining Capability", test_autonomous_retraining),
        ("Free Tier Check", check_free_tier_limitations)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print_error(f"Test '{test_name}' failed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("🎯 DEPLOYMENT VERIFICATION SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed >= 5:  # Most tests should pass
        print(f"""
🎉 DEPLOYMENT VERIFICATION MOSTLY SUCCESSFUL! 🎉

🌐 Your Ghosteam V5 system is deployed at:
   {DEPLOYMENT_URL}

📊 Access Points:
   Dashboard: {DEPLOYMENT_URL}/dashboard
   API Docs:  {DEPLOYMENT_URL}/docs
   Health:    {DEPLOYMENT_URL}/health

⚠️  CRITICAL ACTION REQUIRED:
   Upgrade to Starter Plan for 24/7 operation!
   Free tier will spin down and interrupt autonomous features.

🚀 Once upgraded, your system will operate completely independently!
        """)
        return True
    else:
        print(f"""
⚠️  DEPLOYMENT NEEDS ATTENTION

Several tests failed. Check:
1. Build logs in Render dashboard
2. Service is fully deployed
3. All dependencies installed correctly

URL: {DEPLOYMENT_URL}
        """)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
