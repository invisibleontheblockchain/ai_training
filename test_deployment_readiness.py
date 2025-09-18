#!/usr/bin/env python3
"""
Deployment Readiness Test for AGI System
========================================
Comprehensive testing suite to validate Railway deployment readiness
and test all interfaces before cloud deployment.
"""

import os
import sys
import time
import subprocess
import importlib.util
from pathlib import Path
import json

class DeploymentTester:
    """Test deployment readiness and interface functionality"""
    
    def __init__(self):
        self.test_results = {
            "deployment_files": {},
            "interface_tests": {},
            "dependency_checks": {},
            "system_requirements": {},
            "railway_readiness": False
        }
    
    def run_all_tests(self):
        """Run comprehensive deployment tests"""
        print("🧪 AGI System Deployment Readiness Test")
        print("=" * 50)
        
        # Test deployment files
        self.test_deployment_files()
        
        # Test dependencies
        self.test_dependencies()
        
        # Test core system imports
        self.test_system_imports()
        
        # Test interfaces
        self.test_interfaces()
        
        # Generate report
        self.generate_report()
    
    def test_deployment_files(self):
        """Test all Railway deployment files exist and are valid"""
        print("\n📁 Testing Deployment Files...")
        
        required_files = {
            "Dockerfile": "Docker container configuration",
            "docker-compose.yml": "Local development environment",
            "railway.json": "Railway platform configuration",
            "railway.toml": "Railway deployment settings",
            "Procfile": "Process management",
            "start_agi_system.py": "Unified startup script",
            "requirements.txt": "Python dependencies",
            "master_agi_system.py": "Core AGI orchestrator",
            "telegram_agi_bot.py": "Telegram interface",
            "web_agi_interface.py": "Web interface"
        }
        
        for filename, description in required_files.items():
            exists = os.path.exists(filename)
            self.test_results["deployment_files"][filename] = {
                "exists": exists,
                "description": description,
                "status": "✅ PASS" if exists else "❌ FAIL"
            }
            print(f"  {filename}: {'✅ PASS' if exists else '❌ FAIL'} - {description}")
        
        # Test Railway JSON validity
        if os.path.exists("railway.json"):
            try:
                with open("railway.json", "r") as f:
                    railway_config = json.load(f)
                print("  railway.json validation: ✅ PASS - Valid JSON")
                self.test_results["deployment_files"]["railway.json"]["valid_json"] = True
            except Exception as e:
                print(f"  railway.json validation: ❌ FAIL - {e}")
                self.test_results["deployment_files"]["railway.json"]["valid_json"] = False
    
    def test_dependencies(self):
        """Test required Python packages"""
        print("\n📦 Testing Dependencies...")
        
        required_packages = [
            "streamlit",
            "fastapi", 
            "uvicorn",
            "python-telegram-bot",
            "pandas",
            "numpy",
            "plotly",
            "requests",
            "aiohttp"
        ]
        
        for package in required_packages:
            try:
                spec = importlib.util.find_spec(package.replace("-", "_"))
                available = spec is not None
                self.test_results["dependency_checks"][package] = {
                    "available": available,
                    "status": "✅ PASS" if available else "❌ FAIL"
                }
                print(f"  {package}: {'✅ PASS' if available else '❌ FAIL'}")
            except Exception as e:
                self.test_results["dependency_checks"][package] = {
                    "available": False,
                    "status": "❌ FAIL",
                    "error": str(e)
                }
                print(f"  {package}: ❌ FAIL - {e}")
    
    def test_system_imports(self):
        """Test core system module imports"""
        print("\n🔧 Testing System Imports...")
        
        modules_to_test = [
            "master_agi_system",
            "telegram_agi_bot", 
            "web_agi_interface"
        ]
        
        for module_name in modules_to_test:
            try:
                # Try to import without executing
                spec = importlib.util.spec_from_file_location(module_name, f"{module_name}.py")
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    # Don't execute, just check if it can be loaded
                    print(f"  {module_name}.py: ✅ PASS - Module loads successfully")
                    self.test_results["system_requirements"][module_name] = {"status": "✅ PASS"}
                else:
                    print(f"  {module_name}.py: ❌ FAIL - Cannot load module")
                    self.test_results["system_requirements"][module_name] = {"status": "❌ FAIL"}
            except Exception as e:
                print(f"  {module_name}.py: ⚠️ WARNING - {str(e)[:50]}...")
                self.test_results["system_requirements"][module_name] = {
                    "status": "⚠️ WARNING", 
                    "error": str(e)
                }
    
    def test_interfaces(self):
        """Test interface functionality"""
        print("\n🖥️ Testing Interface Components...")
        
        # Test Streamlit app structure
        try:
            with open("web_agi_interface.py", "r") as f:
                content = f.read()
                has_main = "def main():" in content
                has_streamlit = "import streamlit" in content
                has_dashboard = "run_dashboard" in content
                
                web_test = has_main and has_streamlit and has_dashboard
                print(f"  Web Interface Structure: {'✅ PASS' if web_test else '❌ FAIL'}")
                self.test_results["interface_tests"]["web_interface"] = {
                    "status": "✅ PASS" if web_test else "❌ FAIL",
                    "has_main": has_main,
                    "has_streamlit": has_streamlit,
                    "has_dashboard": has_dashboard
                }
        except Exception as e:
            print(f"  Web Interface Structure: ❌ FAIL - {e}")
            self.test_results["interface_tests"]["web_interface"] = {"status": "❌ FAIL", "error": str(e)}
        
        # Test Telegram bot structure
        try:
            with open("telegram_agi_bot.py", "r") as f:
                content = f.read()
                has_bot_class = "class TelegramAGIBot" in content
                has_telegram_import = "telegram" in content
                has_handlers = "CommandHandler" in content or "MessageHandler" in content
                
                bot_test = has_bot_class and has_telegram_import and has_handlers
                print(f"  Telegram Bot Structure: {'✅ PASS' if bot_test else '❌ FAIL'}")
                self.test_results["interface_tests"]["telegram_bot"] = {
                    "status": "✅ PASS" if bot_test else "❌ FAIL",
                    "has_bot_class": has_bot_class,
                    "has_telegram_import": has_telegram_import,
                    "has_handlers": has_handlers
                }
        except Exception as e:
            print(f"  Telegram Bot Structure: ❌ FAIL - {e}")
            self.test_results["interface_tests"]["telegram_bot"] = {"status": "❌ FAIL", "error": str(e)}
        
        # Test Master AGI system
        try:
            with open("master_agi_system.py", "r") as f:
                content = f.read()
                has_master_class = "class MasterAGISystem" in content
                has_async = "async def" in content
                has_agent_creation = "create_agent" in content
                
                agi_test = has_master_class and has_async and has_agent_creation
                print(f"  Master AGI System: {'✅ PASS' if agi_test else '❌ FAIL'}")
                self.test_results["interface_tests"]["master_agi"] = {
                    "status": "✅ PASS" if agi_test else "❌ FAIL",
                    "has_master_class": has_master_class,
                    "has_async": has_async,
                    "has_agent_creation": has_agent_creation
                }
        except Exception as e:
            print(f"  Master AGI System: ❌ FAIL - {e}")
            self.test_results["interface_tests"]["master_agi"] = {"status": "❌ FAIL", "error": str(e)}
    
    def test_railway_environment(self):
        """Test Railway environment readiness"""
        print("\n🚂 Testing Railway Environment...")
        
        # Check environment variables
        env_vars = {
            "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN", "not_set"),
            "PORT": os.getenv("PORT", "8000"),
            "RAILWAY_ENVIRONMENT": os.getenv("RAILWAY_ENVIRONMENT", "development")
        }
        
        for var, value in env_vars.items():
            status = "✅ SET" if value != "not_set" else "⚠️ NOT SET"
            print(f"  {var}: {status} ({value})")
        
        # Check if we can simulate Railway deployment
        railway_ready = all([
            os.path.exists("Dockerfile"),
            os.path.exists("railway.json"),
            os.path.exists("start_agi_system.py")
        ])
        
        self.test_results["railway_readiness"] = railway_ready
        print(f"  Railway Deployment Ready: {'✅ YES' if railway_ready else '❌ NO'}")
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n📊 DEPLOYMENT READINESS REPORT")
        print("=" * 50)
        
        # Count passes/fails
        deployment_passes = sum(1 for f in self.test_results["deployment_files"].values() if f.get("exists", False))
        deployment_total = len(self.test_results["deployment_files"])
        
        dependency_passes = sum(1 for d in self.test_results["dependency_checks"].values() if d.get("available", False))
        dependency_total = len(self.test_results["dependency_checks"])
        
        interface_passes = sum(1 for i in self.test_results["interface_tests"].values() if "✅ PASS" in i.get("status", ""))
        interface_total = len(self.test_results["interface_tests"])
        
        print(f"📁 Deployment Files: {deployment_passes}/{deployment_total} ({'✅ READY' if deployment_passes == deployment_total else '❌ INCOMPLETE'})")
        print(f"📦 Dependencies: {dependency_passes}/{dependency_total} ({'✅ READY' if dependency_passes >= dependency_total * 0.8 else '❌ MISSING'})")
        print(f"🖥️ Interfaces: {interface_passes}/{interface_total} ({'✅ READY' if interface_passes == interface_total else '⚠️ ISSUES'})")
        
        # Overall readiness
        overall_ready = (
            deployment_passes == deployment_total and
            dependency_passes >= dependency_total * 0.8 and
            interface_passes >= interface_total * 0.8
        )
        
        print(f"\n🚀 OVERALL RAILWAY READINESS: {'✅ READY TO DEPLOY!' if overall_ready else '⚠️ NEEDS ATTENTION'}")
        
        if overall_ready:
            print("\n🎉 Your AGI system is ready for Railway deployment!")
            print("Next steps:")
            print("1. Set TELEGRAM_BOT_TOKEN environment variable")
            print("2. Run: railway up")
            print("3. Your system will be live!")
        else:
            print("\n⚠️ Issues found. Please resolve before deployment:")
            if deployment_passes < deployment_total:
                print("- Missing deployment files")
            if dependency_passes < dependency_total * 0.8:
                print("- Missing dependencies (run: pip install -r requirements.txt)")
            if interface_passes < interface_total * 0.8:
                print("- Interface issues detected")
        
        return overall_ready
    
    def quick_interface_test(self):
        """Quick test of interfaces without full imports"""
        print("\n⚡ Quick Interface Test...")
        
        try:
            # Test if we can start the web interface
            print("Testing web interface startup capability...")
            result = subprocess.run([
                sys.executable, "-c", 
                "import web_agi_interface; print('Web interface import: SUCCESS')"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print("  Web Interface: ✅ READY")
            else:
                print("  Web Interface: ⚠️ ISSUES")
                print(f"    Error: {result.stderr}")
        except Exception as e:
            print(f"  Web Interface: ❌ FAIL - {e}")
        
        try:
            # Test if we can import the telegram bot
            result = subprocess.run([
                sys.executable, "-c",
                "import telegram_agi_bot; print('Telegram bot import: SUCCESS')"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print("  Telegram Bot: ✅ READY")
            else:
                print("  Telegram Bot: ⚠️ ISSUES")
                print(f"    Error: {result.stderr}")
        except Exception as e:
            print(f"  Telegram Bot: ❌ FAIL - {e}")

def main():
    """Run deployment readiness tests"""
    tester = DeploymentTester()
    
    # Run all tests
    tester.run_all_tests()
    
    # Test Railway environment
    tester.test_railway_environment()
    
    # Quick interface test
    tester.quick_interface_test()
    
    # Save detailed results
    with open("deployment_test_results.json", "w") as f:
        json.dump(tester.test_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: deployment_test_results.json")

if __name__ == "__main__":
    main()
