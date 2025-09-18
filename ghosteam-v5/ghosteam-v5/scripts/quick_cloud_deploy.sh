#!/bin/bash

# Ghosteam V5 Quick Cloud Deployment Script
# One-command deployment to any cloud provider
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
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

print_header() {
    echo -e "${BLUE}$1${NC}"
}

print_success() {
    echo -e "${PURPLE}$1${NC}"
}

# ASCII Art Banner
cat << 'EOF'
   ______ __  __ ____   _____ _______ ______          __  __ __      __ _____ 
  / ____// / / // __ \ / ___//_  __/ / ____/         / / / /\ \    / // ____/
 / / __ / /_/ // / / / \__ \  / /   / __/    ______ / / / /  \ \  / //___ \  
/ /_/ // __  // /_/ / ___/ / / /   / /___   /_____// /_/ /    \ \/ / ____/ /  
\____//_/ /_/ \____/ /____/ /_/   /_____/          \____/      \__/ /_____/   

🚀 QUICK CLOUD DEPLOYMENT - MLOps System in Minutes!
EOF

print_header "🌩️ Ghosteam V5 Quick Cloud Deployment"

# Function to detect cloud environment
detect_cloud_provider() {
    if command -v aws &> /dev/null && aws sts get-caller-identity > /dev/null 2>&1; then
        echo "aws"
    elif command -v gcloud &> /dev/null && gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n 1 > /dev/null 2>&1; then
        echo "gcp"
    elif command -v az &> /dev/null && az account show > /dev/null 2>&1; then
        echo "azure"
    else
        echo "none"
    fi
}

# Function to prompt for cloud provider
select_cloud_provider() {
    echo ""
    echo "Select your cloud provider:"
    echo "1) AWS EC2 (Recommended)"
    echo "2) Google Cloud Platform"
    echo "3) Microsoft Azure"
    echo "4) I have my own cloud instance"
    echo ""
    read -p "Enter your choice (1-4): " choice
    
    case $choice in
        1) echo "aws" ;;
        2) echo "gcp" ;;
        3) echo "azure" ;;
        4) echo "manual" ;;
        *) echo "aws" ;;  # Default to AWS
    esac
}

# Main deployment function
main() {
    print_header "🔍 Detecting Cloud Environment"
    
    CLOUD_PROVIDER=$(detect_cloud_provider)
    
    if [ "$CLOUD_PROVIDER" = "none" ]; then
        print_warning "No cloud provider CLI detected or configured"
        CLOUD_PROVIDER=$(select_cloud_provider)
    else
        print_status "✅ Detected cloud provider: $CLOUD_PROVIDER"
        echo -n "Use $CLOUD_PROVIDER for deployment? (y/n): "
        read confirm
        if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
            CLOUD_PROVIDER=$(select_cloud_provider)
        fi
    fi
    
    print_status "Selected cloud provider: $CLOUD_PROVIDER"
    
    case $CLOUD_PROVIDER in
        "aws")
            deploy_aws
            ;;
        "gcp")
            deploy_gcp
            ;;
        "azure")
            deploy_azure
            ;;
        "manual")
            deploy_manual
            ;;
        *)
            print_error "Invalid cloud provider selection"
            exit 1
            ;;
    esac
}

# AWS deployment
deploy_aws() {
    print_header "🚀 Deploying to AWS EC2"
    
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI not found. Installing..."
        curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
        unzip awscliv2.zip
        sudo ./aws/install
    fi
    
    if ! aws sts get-caller-identity > /dev/null 2>&1; then
        print_error "AWS not configured. Please run: aws configure"
        exit 1
    fi
    
    print_status "Starting AWS EC2 deployment..."
    ./scripts/deploy_aws_ec2.sh
    
    # Get the public IP from the AWS deployment
    if [ -f "aws-connection-info.txt" ]; then
        PUBLIC_IP=$(grep "Public IP:" aws-connection-info.txt | cut -d' ' -f3)
        SSH_KEY=$(grep "SSH Command:" aws-connection-info.txt | awk '{print $4}')
        
        print_header "📦 Copying Files and Deploying Application"
        
        # Copy files to instance
        print_status "Copying deployment files..."
        scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -r . ubuntu@"$PUBLIC_IP":~/ghosteam-v5/
        
        # Deploy the application
        print_status "Deploying Ghosteam V5 application..."
        ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no ubuntu@"$PUBLIC_IP" 'cd ghosteam-v5 && ./scripts/deploy_cloud.sh'
        
        show_success_info "$PUBLIC_IP"
    else
        print_error "Failed to get AWS instance information"
        exit 1
    fi
}

# GCP deployment
deploy_gcp() {
    print_header "🚀 Deploying to Google Cloud Platform"
    
    if ! command -v gcloud &> /dev/null; then
        print_error "Google Cloud CLI not found. Please install it first:"
        echo "curl https://sdk.cloud.google.com | bash"
        exit 1
    fi
    
    print_status "Starting GCP deployment..."
    ./scripts/deploy_gcp.sh
    
    # Get the external IP from the GCP deployment
    if [ -f "gcp-connection-info.txt" ]; then
        EXTERNAL_IP=$(grep "External IP:" gcp-connection-info.txt | cut -d' ' -f3)
        INSTANCE_NAME=$(grep "Instance Name:" gcp-connection-info.txt | cut -d' ' -f3)
        ZONE=$(grep "Zone:" gcp-connection-info.txt | cut -d' ' -f2)
        PROJECT=$(grep "Project:" gcp-connection-info.txt | cut -d' ' -f2)
        
        print_header "📦 Copying Files and Deploying Application"
        
        # Copy files to instance
        print_status "Copying deployment files..."
        gcloud compute scp --recurse . "$INSTANCE_NAME":~/ghosteam-v5/ --zone="$ZONE" --project="$PROJECT"
        
        # Deploy the application
        print_status "Deploying Ghosteam V5 application..."
        gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT" --command="cd ghosteam-v5 && ./scripts/deploy_cloud.sh"
        
        show_success_info "$EXTERNAL_IP"
    else
        print_error "Failed to get GCP instance information"
        exit 1
    fi
}

# Azure deployment
deploy_azure() {
    print_header "🚀 Deploying to Microsoft Azure"
    print_warning "Azure deployment script coming soon!"
    print_status "For now, please use manual deployment option"
    deploy_manual
}

# Manual deployment instructions
deploy_manual() {
    print_header "📋 Manual Cloud Deployment Instructions"
    
    echo ""
    echo "To deploy Ghosteam V5 to your existing cloud instance:"
    echo ""
    echo "1. 📤 Copy files to your instance:"
    echo "   scp -r ghosteam-v5/ user@YOUR_PUBLIC_IP:~/"
    echo ""
    echo "2. 🔗 SSH to your instance:"
    echo "   ssh user@YOUR_PUBLIC_IP"
    echo ""
    echo "3. 🚀 Deploy the system:"
    echo "   cd ghosteam-v5"
    echo "   ./scripts/deploy_cloud.sh"
    echo ""
    echo "4. 🌐 Access your system:"
    echo "   API:      http://YOUR_PUBLIC_IP:8080"
    echo "   MLflow:   http://YOUR_PUBLIC_IP:5000"
    echo "   Grafana:  http://YOUR_PUBLIC_IP:3000"
    echo ""
    
    echo -n "Enter your instance public IP (or press Enter to skip): "
    read MANUAL_IP
    
    if [ -n "$MANUAL_IP" ]; then
        show_success_info "$MANUAL_IP"
    fi
}

# Show success information
show_success_info() {
    local PUBLIC_IP=$1
    
    print_header "🎉 DEPLOYMENT COMPLETE!"
    
    cat << EOF

$(print_success "✨ Your Ghosteam V5 MLOps System is now LIVE! ✨")

🌐 Access Your System:
   🚀 API:              http://$PUBLIC_IP:8080
   📓 API Docs:         http://$PUBLIC_IP:8080/docs
   📊 MLflow:           http://$PUBLIC_IP:5000
   📈 Grafana:          http://$PUBLIC_IP:3000 (admin/admin123)
   📉 Prometheus:       http://$PUBLIC_IP:9090
   🍃 Feast:            http://$PUBLIC_IP:6566

🧪 Test Your Deployment:
   curl http://$PUBLIC_IP:8080/health
   curl http://$PUBLIC_IP:8080/models

📊 What's Included:
   ✅ MLflow Tracking Server - Experiment tracking & model registry
   ✅ Feature Store (Feast) - Online & offline feature serving  
   ✅ Model Serving API - FastAPI with auto-scaling
   ✅ Monitoring Stack - Prometheus & Grafana dashboards
   ✅ Databases - PostgreSQL & Redis
   ✅ Sample Data & Models - Ready-to-use examples

🔧 System Management:
   Monitor: ssh to instance and run 'docker-compose -f docker-compose.cloud.yml ps'
   Logs: 'docker-compose -f docker-compose.cloud.yml logs -f api'
   Restart: 'docker-compose -f docker-compose.cloud.yml restart'

📚 Documentation:
   See CLOUD_DEPLOYMENT_GUIDE.md for detailed information

$(print_success "🚀 Ready to build amazing ML applications! 🚀")

EOF
}

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    print_error "Please run this script from the ghosteam-v5 directory"
    exit 1
fi

# Run main function
main "$@"
