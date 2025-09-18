#!/usr/bin/env python3
"""
Railway Deployment Script for Ghosteam V5
Handles the deployment process programmatically
"""

import subprocess
import sys
import time
import json
import os

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

def print_status(message):
    print(f"✅ {message}")

def print_error(message):
    print(f"❌ {message}")

def print_info(message):
    print(f"ℹ️  {message}")

def check_railway_cli():
    """Check if Railway CLI is available and user is logged in"""
    success, output, error = run_command("railway --version")
    if not success:
        print_error("Railway CLI not found. Please install it:")
        print("npm install -g @railway/cli")
        return False
    
    success, output, error = run_command("railway whoami")
    if not success:
        print_error("Not logged in to Railway. Please run: railway login")
        return False
    
    print_status(f"Railway CLI available, logged in as: {output}")
    return True

def deploy_to_railway():
    """Deploy the application to Railway"""
    print("\n🚀 Starting Railway Deployment...")
    
    # Try different service names that might exist
    service_names = ["web", "api", "app", "main", "ghosteam-v5"]
    
    for service_name in service_names:
        print_info(f"Trying to deploy to service: {service_name}")
        success, output, error = run_command(f"railway up --service {service_name} --detach")
        
        if success:
            print_status(f"Successfully deployed to service: {service_name}")
            return True
        else:
            print_info(f"Service {service_name} not found or deployment failed")
    
    # If no specific service works, try creating a new deployment
    print_info("Trying to create new service deployment...")
    success, output, error = run_command("railway up --detach")
    
    if success:
        print_status("Successfully created new deployment")
        return True
    else:
        print_error(f"Deployment failed: {error}")
        return False

def get_deployment_url():
    """Get the deployment URL"""
    print("\n🌐 Getting deployment URL...")
    
    # Try to get domain
    success, output, error = run_command("railway domain")
    if success and "https://" in output:
        url = output.split("https://")[1].split()[0]
        return f"https://{url}"
    
    # If no domain, suggest adding one
    print_info("No domain found. You can add one with: railway domain")
    return None

def test_deployment(url):
    """Test the deployed application"""
    if not url:
        return False
    
    print(f"\n🧪 Testing deployment at: {url}")
    
    try:
        import requests
        
        # Test health endpoint
        response = requests.get(f"{url}/health", timeout=30)
        if response.status_code == 200:
            print_status("Health check passed")
            data = response.json()
            print_info(f"MLflow version: {data.get('mlflow_version', 'unknown')}")
            return True
        else:
            print_error(f"Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Failed to test deployment: {e}")
        return False

def main():
    """Main deployment function"""
    print("🚀 Ghosteam V5 Railway Deployment")
    print("=" * 50)
    
    # Check prerequisites
    if not check_railway_cli():
        return False
    
    # Deploy to Railway
    if not deploy_to_railway():
        return False
    
    # Wait for deployment to complete
    print_info("Waiting for deployment to complete...")
    time.sleep(30)
    
    # Get deployment URL
    url = get_deployment_url()
    
    # Test deployment
    if url:
        test_deployment(url)
    
    # Show success message
    print("\n" + "=" * 50)
    print_status("🎉 Deployment Complete!")
    
    if url:
        print(f"""
🌐 Your Ghosteam V5 MLOps System is now live!

📍 Access URLs:
   🚀 API:              {url}
   📓 API Docs:         {url}/docs
   🔍 Health Check:     {url}/health
   📊 Models:           {url}/models

🧪 Test Commands:
   curl {url}/health
   curl {url}/models

🔧 Management:
   Logs:     railway logs
   Status:   railway status
   Redeploy: railway up

🚀 Your MLOps system is running in the cloud!
        """)
    else:
        print("""
🌐 Deployment completed! Add a domain with:
   railway domain

Then test your deployment:
   railway logs
   railway status
        """)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
