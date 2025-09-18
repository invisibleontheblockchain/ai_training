#!/bin/bash

# Simple Railway Deployment Script for Ghosteam V5
echo "🚀 Deploying Ghosteam V5 MLOps System to Railway..."

# Check if logged in
if ! railway whoami > /dev/null 2>&1; then
    echo "❌ Not logged in to Railway. Please run: railway login"
    exit 1
fi

echo "✅ Logged in to Railway"

# Create a simple railway.json for project configuration
cat > railway.json << EOF
{
  "name": "ghosteam-v5-mlops",
  "description": "Autonomous MLOps System with Continuous Learning",
  "services": {
    "web": {
      "build": {
        "dockerfile": "Dockerfile.railway"
      },
      "deploy": {
        "startCommand": "python -m uvicorn src.minimal_app:app --host 0.0.0.0 --port \$PORT"
      }
    }
  }
}
EOF

echo "✅ Created Railway configuration"

# Try to deploy directly
echo "🚀 Attempting deployment..."
railway up --detach

# Check if deployment was successful
if [ $? -eq 0 ]; then
    echo "✅ Deployment initiated successfully!"
    
    # Wait a moment for deployment to start
    sleep 10
    
    # Try to get domain
    echo "🌐 Checking for domain..."
    railway domain
    
    echo "📊 Checking deployment status..."
    railway status
    
    echo "🎉 Deployment complete! Your system should be available shortly."
    echo ""
    echo "📋 Next steps:"
    echo "1. Check logs: railway logs"
    echo "2. Add domain: railway domain"
    echo "3. Check status: railway status"
    
else
    echo "❌ Deployment failed. Trying alternative approach..."
    
    # Alternative: Try to link to existing project first
    echo "🔗 Trying to link to existing project..."
    
    # List projects to see what's available
    railway list
    
    echo ""
    echo "📋 Manual deployment steps:"
    echo "1. Go to https://railway.app/dashboard"
    echo "2. Create a new project"
    echo "3. Connect your GitHub repository"
    echo "4. Deploy from the web interface"
fi

echo ""
echo "🚀 Your Ghosteam V5 MLOps system deployment is in progress!"
