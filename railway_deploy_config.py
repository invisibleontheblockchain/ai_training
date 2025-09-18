#!/usr/bin/env python3
"""
Railway Deployment Configuration for AGI System
===============================================
Complete deployment setup for the AGI agent ecosystem on Railway.
Handles environment setup, service orchestration, and cloud optimization.
"""

import os
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any

class RailwayDeployment:
    """Handles complete Railway deployment of AGI system"""
    
    def __init__(self):
        self.project_name = "agi-agent-ecosystem"
        self.services = [
            "master-agi-system",
            "telegram-bot",
            "web-interface",
            "api-server"
        ]
        self.setup_deployment_files()
    
    def setup_deployment_files(self):
        """Create all necessary deployment files"""
        self.create_railway_json()
        self.create_dockerfile()
        self.create_docker_compose()
        self.create_requirements_txt()
        self.create_procfile()
        self.create_railway_toml()
        self.create_start_script()
        self.create_environment_template()
    
    def create_railway_json(self):
        """Create railway.json configuration"""
        config = {
            "$schema": "https://railway.app/railway.schema.json",
            "build": {
                "builder": "DOCKERFILE",
                "dockerfilePath": "Dockerfile"
            },
            "deploy": {
                "startCommand": "python start_agi_system.py",
                "healthcheckPath": "/health",
                "healthcheckTimeout": 100,
                "restartPolicyType": "ON_FAILURE",
                "restartPolicyMaxRetries": 3
            }
        }
        
        with open("railway.json", "w") as f:
            json.dump(config, f, indent=2)
    
    def create_dockerfile(self):
        """Create optimized Dockerfile for Railway"""
        dockerfile_content = '''# AGI Agent Ecosystem - Railway Deployment
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    software-properties-common \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY railway_requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r railway_requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p agents logs data

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV RAILWAY_ENVIRONMENT=production

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Start script
CMD ["python", "start_agi_system.py"]
'''
        
        with open("Dockerfile", "w") as f:
            f.write(dockerfile_content)
    
    def create_docker_compose(self):
        """Create docker-compose.yml for local development"""
        compose_content = '''version: '3.8'

services:
  agi-system:
    build: .
    ports:
      - "8000:8000"
      - "8501:8501"
    environment:
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - RAILWAY_ENVIRONMENT=development
      - DATABASE_URL=sqlite:///data/agi_system.db
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./agents:/app/agents
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:
'''
        
        with open("docker-compose.yml", "w") as f:
            f.write(compose_content)
    
    def create_requirements_txt(self):
        """Create comprehensive requirements.txt"""
        requirements = [
            # Core dependencies
            "asyncio",
            "python-telegram-bot>=20.0",
            "streamlit>=1.28.0",
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            
            # Data processing
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "plotly>=5.17.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            
            # Machine Learning (if needed)
            "torch>=2.0.0",
            "transformers>=4.35.0",
            "datasets>=2.14.0",
            "scikit-learn>=1.3.0",
            
            # Database
            "sqlalchemy>=2.0.0",
            "alembic>=1.12.0",
            "redis>=5.0.0",
            
            # Utilities
            "requests>=2.31.0",
            "aiohttp>=3.9.0",
            "python-dotenv>=1.0.0",
            "pydantic>=2.5.0",
            "loguru>=0.7.0",
            
            # Development
            "pytest>=7.4.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0"
        ]
        
        with open("requirements.txt", "w") as f:
            f.write("\n".join(requirements))
    
    def create_railway_requirements(self):
        """Create Railway-optimized requirements"""
        railway_requirements = [
            # Core system - minimal for Railway
            "python-telegram-bot==20.7",
            "streamlit==1.28.1",
            "fastapi==0.104.1",
            "uvicorn[standard]==0.24.0",
            
            # Essential data processing
            "pandas==2.1.3",
            "numpy==1.24.4",
            "plotly==5.17.0",
            
            # Lightweight ML (optional - comment out if not needed)
            # "torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu",
            # "transformers==4.35.2",
            
            # Database and caching
            "redis==5.0.1",
            "python-dotenv==1.0.0",
            "pydantic==2.5.0",
            
            # Utilities
            "requests==2.31.0",
            "aiohttp==3.9.1"
        ]
        
        with open("railway_requirements.txt", "w") as f:
            f.write("\n".join(railway_requirements))
    
    def create_procfile(self):
        """Create Procfile for Railway"""
        procfile_content = '''web: python start_agi_system.py
worker: python master_agi_system.py
'''
        
        with open("Procfile", "w") as f:
            f.write(procfile_content)
    
    def create_railway_toml(self):
        """Create railway.toml configuration"""
        toml_content = '''[build]
builder = "DOCKERFILE"
dockerfilePath = "Dockerfile"

[deploy]
startCommand = "python start_agi_system.py"
healthcheckPath = "/health"
healthcheckTimeout = 100
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 3

[[services]]
name = "agi-system"
source = "."

[services.agi-system.env]
PYTHONPATH = "/app"
PYTHONUNBUFFERED = "1"
RAILWAY_ENVIRONMENT = "production"
'''
        
        with open("railway.toml", "w") as f:
            f.write(toml_content)
    
    def create_start_script(self):
        """Create unified start script for Railway"""
        start_script = '''#!/usr/bin/env python3
"""
AGI System Startup Script for Railway
====================================
Unified startup script that manages all AGI system components.
"""

import asyncio
import os
import sys
import threading
import time
import logging
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

def setup_logging():
    """Setup logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/agi_system.log')
        ]
    )

def check_environment():
    """Check if all required environment variables are set"""
    required_vars = [
        'TELEGRAM_BOT_TOKEN'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in Railway dashboard or .env file")
        return False
    
    return True

def start_web_interface():
    """Start Streamlit web interface"""
    import subprocess
    
    print("🌐 Starting web interface...")
    
    # Start Streamlit in background
    subprocess.Popen([
        'streamlit', 'run', 'web_agi_interface.py',
        '--server.port', '8501',
        '--server.address', '0.0.0.0',
        '--server.headless', 'true',
        '--browser.gatherUsageStats', 'false'
    ])

def start_telegram_bot():
    """Start Telegram bot"""
    print("🤖 Starting Telegram bot...")
    
    try:
        from telegram_agi_bot import TelegramAGIBot
        
        token = os.getenv('TELEGRAM_BOT_TOKEN')
        if token:
            bot = TelegramAGIBot(token)
            
            # Run in separate thread
            def run_bot():
                bot.run()
            
            bot_thread = threading.Thread(target=run_bot, daemon=True)
            bot_thread.start()
            print("✅ Telegram bot started successfully")
        else:
            print("⚠️ Telegram bot token not found, skipping bot startup")
            
    except Exception as e:
        print(f"❌ Failed to start Telegram bot: {e}")

def start_api_server():
    """Start FastAPI server"""
    print("🚀 Starting API server...")
    
    try:
        import uvicorn
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from master_agi_system import get_master_agi
        
        app = FastAPI(title="AGI Agent API", version="1.0.0")
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Get AGI system instance
        agi_system = get_master_agi()
        
        @app.get("/")
        async def root():
            return {"message": "AGI Agent System API", "status": "running"}
        
        @app.get("/health")
        async def health_check():
            """Health check endpoint for Railway"""
            try:
                system_status = agi_system.get_system_status()
                return {
                    "status": "healthy",
                    "timestamp": time.time(),
                    "agents": system_status['total_agents'],
                    "uptime": "running"
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/agents")
        async def get_agents():
            """Get all active agents"""
            try:
                agents = agi_system.get_active_agents()
                return {"agents": agents}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/agents")
        async def create_agent(agent_data: dict):
            """Create new agent"""
            try:
                agent_id = await agi_system.create_agent(
                    agent_data.get('type'),
                    agent_data.get('purpose'),
                    agent_data.get('specialization')
                )
                return {"agent_id": agent_id, "status": "created"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/system/status")
        async def get_system_status():
            """Get system status"""
            try:
                status = agi_system.get_system_status()
                return status
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Run FastAPI server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=int(os.getenv("PORT", 8000)),
            log_level="info"
        )
        
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")

def start_master_agi():
    """Start master AGI system"""
    print("🧠 Starting Master AGI System...")
    
    try:
        from master_agi_system import MasterAGISystem
        
        # Create and start AGI system in background
        def run_agi():
            agi_system = MasterAGISystem()
            asyncio.run(agi_system.start_system())
        
        agi_thread = threading.Thread(target=run_agi, daemon=True)
        agi_thread.start()
        print("✅ Master AGI System started")
        
    except Exception as e:
        print(f"❌ Failed to start Master AGI System: {e}")

def main():
    """Main startup function"""
    print("🚀 Starting AGI Agent Ecosystem...")
    print("=" * 50)
    
    # Setup logging
    setup_logging()
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("agents", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Start components
    try:
        # Start master AGI system first
        start_master_agi()
        time.sleep(2)  # Give it time to initialize
        
        # Start Telegram bot
        start_telegram_bot()
        time.sleep(1)
        
        # Start web interface
        start_web_interface()
        time.sleep(1)
        
        print("✅ All systems started successfully!")
        print("🌐 Web interface: http://localhost:8501")
        print("🚀 API server: http://localhost:8000")
        print("📱 Telegram bot: Active and listening")
        
        # Start API server (this will run in main thread)
        start_api_server()
        
    except KeyboardInterrupt:
        print("\\n⏹️ System stopped by user")
    except Exception as e:
        print(f"\\n❌ System startup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        with open("start_agi_system.py", "w") as f:
            f.write(start_script)
    
    def create_environment_template(self):
        """Create .env template"""
        env_template = '''# AGI Agent Ecosystem Environment Configuration
# =============================================

# Required: Telegram Bot Token (get from @BotFather)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here

# Optional: Database URL (defaults to SQLite)
# DATABASE_URL=postgresql://username:password@localhost/agi_system

# Optional: Redis URL for caching
# REDIS_URL=redis://localhost:6379

# Optional: Model configurations
# OPENAI_API_KEY=your_openai_key_here
# ANTHROPIC_API_KEY=your_anthropic_key_here

# Railway Environment (automatically set by Railway)
# RAILWAY_ENVIRONMENT=production
# PORT=8000

# Development settings
# DEBUG=false
# LOG_LEVEL=INFO
'''
        
        with open(".env.example", "w") as f:
            f.write(env_template)
    
    def create_deployment_guide(self):
        """Create deployment guide"""
        guide = '''# AGI Agent Ecosystem - Railway Deployment Guide

## Quick Deploy to Railway

### 1. Prerequisites
- Railway account (https://railway.app)
- Telegram Bot Token from @BotFather
- Git repository with your code

### 2. One-Click Deploy
[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template/your-template-id)

### 3. Manual Deployment

1. **Fork/Clone this repository**
   ```bash
   git clone <your-repo-url>
   cd agi-agent-ecosystem
   ```

2. **Install Railway CLI**
   ```bash
   npm install -g @railway/cli
   railway login
   ```

3. **Create Railway Project**
   ```bash
   railway init
   railway add
   ```

4. **Set Environment Variables**
   ```bash
   railway variables set TELEGRAM_BOT_TOKEN=your_bot_token_here
   ```

5. **Deploy**
   ```bash
   railway up
   ```

### 4. Environment Variables

Required:
- `TELEGRAM_BOT_TOKEN`: Your Telegram bot token

Optional:
- `DATABASE_URL`: PostgreSQL database URL
- `REDIS_URL`: Redis cache URL
- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key

### 5. Accessing Your Deployed System

After deployment, you'll get URLs for:
- **Web Interface**: `https://your-app.railway.app:8501`
- **API**: `https://your-app.railway.app:8000`
- **Health Check**: `https://your-app.railway.app:8000/health`

### 6. Telegram Bot Setup

1. Get your Railway app URL
2. Set webhook (optional): 
   ```
   https://api.telegram.org/bot<TOKEN>/setWebhook?url=https://your-app.railway.app/webhook
   ```

### 7. Usage

1. **Telegram**: Message your bot directly
2. **Web Interface**: Visit the web URL
3. **API**: Use the REST API endpoints

### 8. Monitoring

- Health check: `/health` endpoint
- Logs: Railway dashboard
- Metrics: Built-in Railway monitoring

### 9. Scaling

Railway automatically scales based on usage. You can also:
- Enable autoscaling in Railway dashboard
- Set resource limits
- Add additional services

### 10. Cost Optimization

- Use Railway's free tier for development
- Enable hibernation for inactive periods
- Monitor usage in Railway dashboard

## Troubleshooting

### Common Issues:

1. **Bot not responding**
   - Check TELEGRAM_BOT_TOKEN is set correctly
   - Verify bot is started: check logs

2. **Web interface not loading**
   - Check port 8501 is exposed
   - Verify Streamlit is running: check logs

3. **API errors**
   - Check /health endpoint
   - Review application logs

### Getting Help:

- Check Railway logs: `railway logs`
- Discord: Railway community
- GitHub Issues: This repository

## Advanced Configuration

### Custom Domains
```bash
railway domain add your-domain.com
```

### Multiple Environments
```bash
railway environment create staging
railway environment create production
```

### Database Setup
```bash
railway add postgresql
railway variables set DATABASE_URL=$DATABASE_URL
```

### Redis Cache
```bash
railway add redis
railway variables set REDIS_URL=$REDIS_URL
```
'''
        
        with open("DEPLOYMENT_GUIDE.md", "w") as f:
            f.write(guide)
    
    def create_railway_template(self):
        """Create Railway template configuration"""
        template = {
            "name": "AGI Agent Ecosystem",
            "description": "Self-replicating AI agent system with Telegram bot and web interface",
            "repository": "https://github.com/your-username/agi-agent-ecosystem",
            "services": [
                {
                    "name": "agi-system",
                    "runtime": "docker",
                    "buildCommand": "docker build -t agi-system .",
                    "startCommand": "python start_agi_system.py",
                    "envVars": {
                        "TELEGRAM_BOT_TOKEN": {
                            "description": "Telegram Bot Token from @BotFather",
                            "required": True
                        },
                        "RAILWAY_ENVIRONMENT": {
                            "description": "Railway Environment",
                            "default": "production"
                        }
                    },
                    "domains": [
                        {
                            "name": "${{RAILWAY_STATIC_URL}}"
                        }
                    ]
                }
            ],
            "plugins": [
                {
                    "name": "postgresql",
                    "plan": "hobby"
                },
                {
                    "name": "redis",
                    "plan": "hobby"
                }
            ]
        }
        
        with open("railway-template.json", "w") as f:
            json.dump(template, f, indent=2)
    
    def deploy_to_railway(self):
        """Deploy the system to Railway"""
        print("🚀 Deploying AGI System to Railway...")
        
        try:
            # Check if Railway CLI is installed
            subprocess.run(["railway", "--version"], check=True, capture_output=True)
            print("✅ Railway CLI found")
            
            # Initialize Railway project
            print("📝 Initializing Railway project...")
            subprocess.run(["railway", "init"], check=True)
            
            # Add service
            print("🔧 Adding service...")
            subprocess.run(["railway", "add"], check=True)
            
            # Set environment variables
            print("⚙️ Setting environment variables...")
            bot_token = input("Enter your Telegram Bot Token: ")
            if bot_token:
                subprocess.run(["railway", "variables", "set", f"TELEGRAM_BOT_TOKEN={bot_token}"], check=True)
            
            # Deploy
            print("🚀 Deploying to Railway...")
            result = subprocess.run(["railway", "up"], check=True, capture_output=True, text=True)
            
            print("✅ Deployment successful!")
            print("🌐 Your AGI system is now live on Railway!")
            
            # Get deployment URL
            try:
                status_result = subprocess.run(["railway", "status"], capture_output=True, text=True)
                print(f"📊 Deployment status:\n{status_result.stdout}")
            except:
                pass
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Deployment failed: {e}")
            print("Please install Railway CLI: npm install -g @railway/cli")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

def main():
    """Main deployment setup function"""
    print("🧠 AGI Agent Ecosystem - Railway Deployment Setup")
    print("=" * 60)
    
    deployer = RailwayDeployment()
    
    # Create all deployment files
    print("📝 Creating deployment files...")
    deployer.create_railway_requirements()
    deployer.create_deployment_guide()
    deployer.create_railway_template()
    
    print("✅ All deployment files created!")
    print("\nFiles created:")
    print("- Dockerfile")
    print("- docker-compose.yml")
    print("- requirements.txt")
    print("- railway_requirements.txt")
    print("- railway.json")
    print("- railway.toml")
    print("- Procfile")
    print("- start_agi_system.py")
    print("- .env.example")
    print("- DEPLOYMENT_GUIDE.md")
    print("- railway-template.json")
    
    print("\n🚀 Ready for Railway deployment!")
    print("\nNext steps:")
    print("1. Set TELEGRAM_BOT_TOKEN environment variable")
    print("2. Run: railway init && railway up")
    print("3. Or use the deployment guide: DEPLOYMENT_GUIDE.md")
    
    # Ask if user wants to deploy now
    deploy_now = input("\nDeploy to Railway now? (y/n): ").lower().strip()
    if deploy_now == 'y':
        deployer.deploy_to_railway()

if __name__ == "__main__":
    main()
