#!/bin/bash

echo "🚨 EMERGENCY CLOUD DEPLOYMENT FOR TRAVEL"
echo "========================================"

# Check Railway login
if ! railway whoami > /dev/null 2>&1; then
    echo "❌ Not logged in to Railway"
    echo "Please run: railway login"
    exit 1
fi

echo "✅ Railway CLI available"

# Try to deploy using different methods
echo "🚀 Attempting Railway deployment..."

# Method 1: Direct deployment
echo "Method 1: Direct deployment"
if railway up --detach; then
    echo "✅ Deployment successful!"
    sleep 10
    railway domain
    exit 0
fi

# Method 2: Create service first
echo "Method 2: Creating service"
if echo "web" | railway service create; then
    echo "✅ Service created"
    if railway up --detach; then
        echo "✅ Deployment successful!"
        sleep 10
        railway domain
        exit 0
    fi
fi

# Method 3: Manual instructions
echo "❌ Automatic deployment failed"
echo ""
echo "🔧 MANUAL DEPLOYMENT REQUIRED:"
echo "1. Go to: https://railway.app/dashboard"
echo "2. Create new project from GitHub"
echo "3. Select your repository"
echo "4. Use Dockerfile.autonomous"
echo "5. Set environment variables:"
echo "   PYTHONPATH=/app"
echo "   ENVIRONMENT=production"
echo "   PORT=8080"
echo ""
echo "⚠️  CRITICAL: Deploy before traveling!"
echo "Your local system will NOT work when computer is off!"
