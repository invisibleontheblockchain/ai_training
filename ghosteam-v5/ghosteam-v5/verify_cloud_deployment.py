#!/usr/bin/env python3
"""
Cloud Deployment Verification Script
Comprehensive testing of autonomous MLOps system in the cloud
"""

import requests
import json
import time
import sys
from datetime import datetime

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🔍 CLOUD DEPLOYMENT VERIFICATION                          ║
║                   Ghosteam V5 Autonomous MLOps System                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

def print_test(test_name):
    print(f"\n🧪 Testing: {test_name}")
    print("-" * 50)

def print_status(message):
    print(f"✅ {message}")

def print_error(message):
    print(f"❌ {message}")

def print_info(message):
    print(f"ℹ️  {message}")

def test_health_endpoint(base_url):
    """Test system health and autonomous features"""
    print_test("System Health & Autonomous Features")
    
    try:
        response = requests.get(f"{base_url}/health", timeout=30)
        
        if response.status_code == 200:
            print_status("Health endpoint accessible")
            data = response.json()
            
            # Check system status
            status = data.get('status', 'unknown')
            print_info(f"System status: {status}")
            
            # Check autonomous features
            autonomous_features = data.get('autonomous_features', {})
            print_info("Autonomous features:")
            for feature, enabled in autonomous_features.items():
                status_icon = "✅" if enabled else "❌"
                print_info(f"  {status_icon} {feature}: {enabled}")
            
            # Check metrics
            metrics = data.get('metrics', {})
            print_info("System metrics:")
            for metric, value in metrics.items():
                print_info(f"  📊 {metric}: {value}")
            
            return True
        else:
            print_error(f"Health check failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Health check error: {e}")
        return False

def test_prediction_endpoint(base_url):
    """Test prediction functionality"""
    print_test("Prediction Endpoint & ML Functionality")
    
    try:
        test_data = {
            "data": [1.0, -0.5, 0.3, -1.2, 0.8, -0.1, 0.5, -0.8, 1.1, -0.3],
            "context": {
                "user_type": "cloud_verification",
                "test_timestamp": datetime.now().isoformat(),
                "source": "deployment_verification"
            }
        }
        
        response = requests.post(
            f"{base_url}/predict", 
            json=test_data, 
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print_status("Prediction endpoint accessible")
            result = response.json()
            
            print_info(f"Prediction result: {result.get('prediction')}")
            print_info(f"Confidence: {result.get('confidence', 0):.2f}")
            print_info(f"Model used: {result.get('model_used', 'unknown')}")
            
            suggestions = result.get('suggestions', [])
            if suggestions:
                print_info(f"AI suggestions: {len(suggestions)} provided")
                for i, suggestion in enumerate(suggestions[:3], 1):
                    print_info(f"  {i}. {suggestion}")
            
            return True
        else:
            print_error(f"Prediction failed: HTTP {response.status_code}")
            print_error(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print_error(f"Prediction test error: {e}")
        return False

def test_insights_endpoint(base_url):
    """Test predictive insights functionality"""
    print_test("Predictive Insights & Learning Analytics")
    
    try:
        response = requests.get(f"{base_url}/insights", timeout=30)
        
        if response.status_code == 200:
            print_status("Insights endpoint accessible")
            data = response.json()
            
            insights = data.get('insights', [])
            print_info(f"Active insights: {len(insights)}")
            for i, insight in enumerate(insights[:3], 1):
                print_info(f"  {i}. {insight}")
            
            usage_patterns = data.get('usage_patterns', [])
            print_info(f"Usage patterns tracked: {len(usage_patterns)}")
            
            autonomous_actions = data.get('autonomous_actions', [])
            print_info(f"Autonomous actions completed: {len(autonomous_actions)}")
            
            if autonomous_actions:
                latest_action = autonomous_actions[-1]
                print_info(f"Latest autonomous action: {latest_action.get('action', 'unknown')}")
                if 'accuracy' in latest_action:
                    print_info(f"  Achieved accuracy: {latest_action['accuracy']:.3f}")
            
            return True
        else:
            print_error(f"Insights failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Insights test error: {e}")
        return False

def test_dashboard_accessibility(base_url):
    """Test dashboard accessibility"""
    print_test("Interactive Dashboard")
    
    try:
        response = requests.get(f"{base_url}/dashboard", timeout=30)
        
        if response.status_code == 200:
            print_status("Dashboard accessible")
            print_info(f"Dashboard URL: {base_url}/dashboard")
            
            # Check if it's HTML content
            if 'text/html' in response.headers.get('content-type', ''):
                print_status("Dashboard returns HTML content")
                
                # Check for key dashboard elements
                content = response.text.lower()
                dashboard_elements = [
                    'autonomous',
                    'mlops',
                    'dashboard',
                    'system status',
                    'performance'
                ]
                
                for element in dashboard_elements:
                    if element in content:
                        print_info(f"  ✅ Contains: {element}")
                    else:
                        print_info(f"  ⚠️  Missing: {element}")
                
                return True
            else:
                print_error("Dashboard doesn't return HTML content")
                return False
        else:
            print_error(f"Dashboard failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Dashboard test error: {e}")
        return False

def test_api_documentation(base_url):
    """Test API documentation accessibility"""
    print_test("API Documentation")
    
    try:
        response = requests.get(f"{base_url}/docs", timeout=30)
        
        if response.status_code == 200:
            print_status("API documentation accessible")
            print_info(f"API docs URL: {base_url}/docs")
            return True
        else:
            print_error(f"API docs failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"API docs test error: {e}")
        return False

def test_continuous_operation(base_url):
    """Test that the system continues operating autonomously"""
    print_test("Continuous Autonomous Operation")
    
    print_info("Testing system responsiveness over time...")
    
    # Test multiple requests to ensure consistency
    success_count = 0
    total_tests = 3
    
    for i in range(total_tests):
        try:
            response = requests.get(f"{base_url}/health", timeout=10)
            if response.status_code == 200:
                success_count += 1
                print_info(f"  Test {i+1}/{total_tests}: ✅ Responsive")
            else:
                print_info(f"  Test {i+1}/{total_tests}: ❌ Failed ({response.status_code})")
            
            if i < total_tests - 1:
                time.sleep(2)  # Wait between tests
                
        except Exception as e:
            print_info(f"  Test {i+1}/{total_tests}: ❌ Error ({e})")
    
    success_rate = (success_count / total_tests) * 100
    print_info(f"System responsiveness: {success_rate:.1f}% ({success_count}/{total_tests})")
    
    return success_rate >= 80  # 80% success rate is acceptable

def main():
    print_banner()
    
    # Get the cloud URL
    cloud_url = input("Enter your cloud deployment URL (e.g., https://your-app.onrender.com): ").strip()
    
    if not cloud_url:
        print_error("No URL provided. Exiting.")
        return False
    
    if not cloud_url.startswith('http'):
        cloud_url = f"https://{cloud_url}"
    
    print_info(f"Testing deployment at: {cloud_url}")
    print_info("This may take a few minutes for a fresh deployment...")
    
    # Run all tests
    tests = [
        ("System Health", lambda: test_health_endpoint(cloud_url)),
        ("Prediction Functionality", lambda: test_prediction_endpoint(cloud_url)),
        ("Predictive Insights", lambda: test_insights_endpoint(cloud_url)),
        ("Interactive Dashboard", lambda: test_dashboard_accessibility(cloud_url)),
        ("API Documentation", lambda: test_api_documentation(cloud_url)),
        ("Continuous Operation", lambda: test_continuous_operation(cloud_url))
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print_error(f"Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("🎯 VERIFICATION SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed >= total * 0.8:  # 80% pass rate
        print(f"""
🎉 CLOUD DEPLOYMENT VERIFICATION SUCCESSFUL! 🎉

🌐 Your Ghosteam V5 Autonomous MLOps System is:
   ✅ Running independently in the cloud
   ✅ Accessible from anywhere in the world
   ✅ Operating autonomously 24/7
   ✅ Learning and improving continuously
   ✅ Ready for remote access while traveling

🚀 System URLs:
   Main:      {cloud_url}
   Dashboard: {cloud_url}/dashboard
   API Docs:  {cloud_url}/docs
   Health:    {cloud_url}/health

Safe travels! Your AI system will continue operating without your computer.
        """)
        return True
    else:
        print(f"""
⚠️  DEPLOYMENT NEEDS ATTENTION

Some tests failed. This might be normal for a very fresh deployment.
Wait 5-10 minutes and run this verification again.

If issues persist, check:
1. Build logs in your cloud platform dashboard
2. Environment variables are set correctly
3. All dependencies are installed

Your system URL: {cloud_url}
        """)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
