#!/bin/bash

# Ghosteam V5 Google Cloud Platform Deployment Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_header "🚀 Ghosteam V5 Google Cloud Platform Deployment"

# Configuration
MACHINE_TYPE="${MACHINE_TYPE:-e2-standard-4}"  # 4 vCPU, 16GB RAM
ZONE="${GCP_ZONE:-us-central1-a}"
PROJECT_ID="${GCP_PROJECT_ID}"
INSTANCE_NAME="ghosteam-v5-instance"

# Check gcloud CLI
if ! command -v gcloud &> /dev/null; then
    print_error "Google Cloud CLI is not installed. Please install it first:"
    echo "curl https://sdk.cloud.google.com | bash"
    echo "exec -l \$SHELL"
    echo "gcloud init"
    exit 1
fi

# Check if authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n 1 > /dev/null; then
    print_error "Not authenticated with Google Cloud. Please run: gcloud auth login"
    exit 1
fi

# Get or set project ID
if [ -z "$PROJECT_ID" ]; then
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
    if [ -z "$PROJECT_ID" ]; then
        print_error "No project ID set. Please run: gcloud config set project YOUR_PROJECT_ID"
        exit 1
    fi
fi

print_status "✅ Using Google Cloud project: $PROJECT_ID"

# Enable required APIs
print_header "Enabling Required APIs"
gcloud services enable compute.googleapis.com --project="$PROJECT_ID"
print_status "✅ Compute Engine API enabled"

# Create firewall rules
print_header "Setting Up Firewall Rules"
if ! gcloud compute firewall-rules describe ghosteam-v5-allow-http --project="$PROJECT_ID" > /dev/null 2>&1; then
    print_status "Creating firewall rules..."
    
    gcloud compute firewall-rules create ghosteam-v5-allow-http \
        --allow tcp:80,tcp:443,tcp:22,tcp:5000,tcp:8080,tcp:3000,tcp:9090,tcp:6566 \
        --source-ranges 0.0.0.0/0 \
        --description "Allow HTTP/HTTPS and Ghosteam V5 ports" \
        --project="$PROJECT_ID"
    
    print_status "✅ Firewall rules created"
else
    print_status "✅ Firewall rules already exist"
fi

# Create startup script
cat > startup-script.sh << 'EOF'
#!/bin/bash
set -e

# Update system
apt-get update
apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker $(whoami)

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install Python and tools
apt-get install -y python3 python3-pip python3-venv git curl

# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Signal completion
touch /home/$(whoami)/setup-complete
EOF

# Check if instance exists
print_header "Checking Instance Status"
if gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT_ID" > /dev/null 2>&1; then
    print_status "✅ Instance already exists: $INSTANCE_NAME"
    INSTANCE_STATUS=$(gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT_ID" --format="value(status)")
    
    if [ "$INSTANCE_STATUS" != "RUNNING" ]; then
        print_status "Starting existing instance..."
        gcloud compute instances start "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT_ID"
    fi
else
    print_status "Creating new instance..."
    
    gcloud compute instances create "$INSTANCE_NAME" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --network-interface=network-tier=PREMIUM,subnet=default \
        --metadata-from-file startup-script=startup-script.sh \
        --maintenance-policy=MIGRATE \
        --provisioning-model=STANDARD \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --tags=http-server,https-server \
        --create-disk=auto-delete=yes,boot=yes,device-name="$INSTANCE_NAME",image=projects/ubuntu-os-cloud/global/images/family/ubuntu-2204-lts,mode=rw,size=50,type=projects/"$PROJECT_ID"/zones/"$ZONE"/diskTypes/pd-standard \
        --no-shielded-secure-boot \
        --shielded-vtpm \
        --shielded-integrity-monitoring \
        --reservation-affinity=any \
        --project="$PROJECT_ID"
    
    print_status "✅ Instance created: $INSTANCE_NAME"
fi

# Get external IP
print_status "Getting instance IP..."
EXTERNAL_IP=$(gcloud compute instances describe "$INSTANCE_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT_ID" \
    --format="value(networkInterfaces[0].accessConfigs[0].natIP)")

print_status "✅ Instance external IP: $EXTERNAL_IP"

# Wait for SSH to be available
print_status "Waiting for SSH to be available..."
while ! gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT_ID" --command="echo 'SSH ready'" > /dev/null 2>&1; do
    sleep 5
done

print_status "✅ SSH is available"

# Wait for setup to complete
print_status "Waiting for instance setup to complete..."
while ! gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT_ID" --command="test -f /home/\$(whoami)/setup-complete" > /dev/null 2>&1; do
    echo -n "."
    sleep 10
done
echo ""

print_header "🎉 Google Cloud Instance Ready!"

echo ""
echo "📊 Instance Details:"
echo "   Instance Name: $INSTANCE_NAME"
echo "   External IP:   $EXTERNAL_IP"
echo "   Zone:          $ZONE"
echo "   Project:       $PROJECT_ID"
echo ""

echo "🔗 SSH Access:"
echo "   gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID"
echo ""

echo "📋 Next Steps:"
echo "1. Copy deployment files to instance:"
echo "   gcloud compute scp --recurse ghosteam-v5/ $INSTANCE_NAME:~/ --zone=$ZONE --project=$PROJECT_ID"
echo ""
echo "2. Deploy Ghosteam V5:"
echo "   gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID"
echo "   cd ghosteam-v5"
echo "   ./scripts/deploy_cloud.sh"
echo ""

echo "🌐 Future Access Points (after deployment):"
echo "   API:      http://$EXTERNAL_IP:8080"
echo "   MLflow:   http://$EXTERNAL_IP:5000"
echo "   Grafana:  http://$EXTERNAL_IP:3000"
echo ""

# Save connection info
cat > gcp-connection-info.txt << EOF
# Ghosteam V5 GCP Connection Information
Instance Name: $INSTANCE_NAME
External IP: $EXTERNAL_IP
Zone: $ZONE
Project: $PROJECT_ID

# SSH Command:
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID

# Copy Files:
gcloud compute scp --recurse ghosteam-v5/ $INSTANCE_NAME:~/ --zone=$ZONE --project=$PROJECT_ID

# Access URLs (after deployment):
API: http://$EXTERNAL_IP:8080
MLflow: http://$EXTERNAL_IP:5000
Grafana: http://$EXTERNAL_IP:3000
Prometheus: http://$EXTERNAL_IP:9090
EOF

print_status "✅ Connection info saved to gcp-connection-info.txt"
print_status "🚀 Ready to deploy Ghosteam V5!"
