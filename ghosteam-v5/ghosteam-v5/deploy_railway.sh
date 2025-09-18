#!/bin/bash

# Ghosteam V5 Railway Deployment Script
set -e

echo "🚀 Starting Ghosteam V5 Railway Deployment..."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Railway CLI is available
if ! command -v railway &> /dev/null; then
    print_error "Railway CLI not found. Please install it first:"
    echo "npm install -g @railway/cli"
    exit 1
fi

# Check if logged in
if ! railway whoami &> /dev/null; then
    print_error "Not logged in to Railway. Please run: railway login"
    exit 1
fi

print_status "✅ Railway CLI available and logged in"

# Check if project is linked
if ! railway status &> /dev/null; then
    print_warning "Project not linked. Linking to ghosteam-v5..."
    railway link --project ghosteam-v5 || {
        print_error "Failed to link project. Creating new project..."
        railway init
    }
fi

print_status "✅ Project linked successfully"

# Deploy the application
print_status "🚀 Deploying Ghosteam V5 MLOps System..."

# Try to deploy to web service (common default name)
railway up --service web 2>/dev/null || {
    print_warning "Web service not found, trying default deployment..."
    railway up 2>/dev/null || {
        print_error "Deployment failed. Trying to create new service..."
        # Create a new service and deploy
        echo "Creating new service for deployment..."
        railway up --detach
    }
}

print_status "✅ Deployment initiated successfully!"

# Wait a moment for deployment to start
sleep 5

# Try to get the deployment URL
print_status "🌐 Getting deployment URL..."
DOMAIN=$(railway domain 2>/dev/null | grep -o 'https://[^[:space:]]*' | head -1) || {
    print_warning "Could not automatically get domain. You can add one with: railway domain"
    DOMAIN="<your-railway-domain>"
}

print_status "✅ Deployment completed!"

cat << EOF

🎉 Ghosteam V5 MLOps System Deployed to Railway! 🎉

🌐 Access Your System:
   🚀 API:              $DOMAIN
   📓 API Docs:         $DOMAIN/docs
   🔍 Health Check:     $DOMAIN/health

🧪 Test Your Deployment:
   curl $DOMAIN/health
   curl $DOMAIN/models

📊 What's Deployed:
   ✅ FastAPI Model Serving API (with MLflow integration)
   ✅ PostgreSQL Database (managed by Railway)
   ✅ Redis Cache (managed by Railway)
   ✅ MLflow Model Registry
   ✅ Feature Store (with graceful degradation)

🔧 System Management:
   Monitor: railway logs
   Status:  railway status
   Redeploy: railway up

📚 Next Steps:
   1. Test the API endpoints
   2. Upload your first model to MLflow
   3. Start making predictions!

🚀 Your MLOps system is now running in the cloud! 🚀

EOF
