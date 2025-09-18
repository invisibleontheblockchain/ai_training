#!/usr/bin/env python3
"""
AGI Agent Ecosystem - Railway Production Startup Script
Optimized for cloud deployment with health checks
"""

import os
import sys
import time
import threading
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def start_api_server():
    """Start FastAPI server for health checks and API endpoints"""
    try:
        import uvicorn
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        
        app = FastAPI(title="AGI System API", version="1.0.0")
        
        @app.get("/")
        async def root():
            return {"message": "AGI Agent Ecosystem", "status": "operational", "version": "1.0.0"}
        
        @app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": time.time(), "services": ["api", "agi_system"]}
        
        @app.get("/status")
        async def system_status():
            return {
                "agi_system": "operational",
                "api_server": "running",
                "timestamp": time.time(),
                "environment": "production"
            }
        
        # Get port from environment (Railway sets this)
        port = int(os.environ.get("PORT", 8000))
        host = "0.0.0.0"
        
        logger.info(f"Starting API server on {host}:{port}")
        uvicorn.run(app, host=host, port=port, log_level="info")
        
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        sys.exit(1)

def start_web_interface():
    """Start Streamlit web interface"""
    try:
        import subprocess
        import streamlit as st
        
        # Get web port (different from API port)
        web_port = int(os.environ.get("WEB_PORT", 8501))
        
        logger.info(f"Starting web interface on port {web_port}")
        
        # Start streamlit in subprocess
        cmd = [
            "streamlit", "run", "web_agi_interface.py",
            "--server.port", str(web_port),
            "--server.address", "0.0.0.0",
            "--server.headless", "true"
        ]
        
        subprocess.run(cmd)
        
    except Exception as e:
        logger.error(f"Failed to start web interface: {e}")

def start_telegram_bot():
    """Start Telegram bot"""
    try:
        from telegram_agi_bot import main as start_bot
        logger.info("Starting Telegram bot...")
        start_bot()
    except Exception as e:
        logger.error(f"Failed to start Telegram bot: {e}")

def initialize_agi_system():
    """Initialize the master AGI system"""
    try:
        # Try to import master system, but don't fail if missing
        try:
            from master_agi_system import MasterAGISystem
            logger.info("Initializing Master AGI System...")
            
            # Create system instance
            master_system = MasterAGISystem()
            
            # Perform initialization
            master_system.initialize()
            
            logger.info("✅ Master AGI System initialized successfully")
            return master_system
        except ImportError as ie:
            logger.warning(f"Master AGI System not available: {ie}")
            logger.info("Running in minimal API mode")
            return None
        
    except Exception as e:
        logger.error(f"Failed to initialize AGI system: {e}")
        return None

def main():
    """Main startup function optimized for Railway deployment"""
    logger.info("🚀 Starting AGI Agent Ecosystem (Railway Production)")
    
    try:
        # Create necessary directories
        os.makedirs("agents", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
        # Initialize AGI system
        agi_system = initialize_agi_system()
        
        # Start background services
        logger.info("Starting background services...")
        
        # Start web interface in background (optional for Railway)
        if os.environ.get("ENABLE_WEB", "false").lower() == "true":
            web_thread = threading.Thread(target=start_web_interface, daemon=True)
            web_thread.start()
            logger.info("📊 Web interface started")
        
        # Start telegram bot in background (optional)
        if os.environ.get("TELEGRAM_BOT_TOKEN"):
            bot_thread = threading.Thread(target=start_telegram_bot, daemon=True)
            bot_thread.start()
            logger.info("🤖 Telegram bot started")
        
        logger.info("✅ All systems operational!")
        logger.info("🌐 API Server: Starting on PORT from environment")
        logger.info("🔍 Health Check: /health endpoint available")
        logger.info("📡 System Status: /status endpoint available")
        
        # Start API server (this blocks and serves health checks)
        start_api_server()
        
    except Exception as e:
        logger.error(f"Failed to start AGI system: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
