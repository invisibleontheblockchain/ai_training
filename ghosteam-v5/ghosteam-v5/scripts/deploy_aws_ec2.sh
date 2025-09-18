#!/bin/bash

# Ghosteam V5 AWS EC2 Cloud Deployment Script
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

print_header "🚀 Ghosteam V5 AWS EC2 Cloud Deployment"

# Configuration
INSTANCE_TYPE="${INSTANCE_TYPE:-t3.xlarge}"  # 4 vCPU, 16GB RAM
REGION="${AWS_REGION:-us-west-2}"
KEY_NAME="${KEY_NAME:-ghosteam-v5-key}"
SECURITY_GROUP_NAME="ghosteam-v5-sg"
INSTANCE_NAME="ghosteam-v5-instance"

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    print_error "AWS CLI is not installed. Please install it first:"
    echo "curl 'https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip' -o 'awscliv2.zip'"
    echo "unzip awscliv2.zip && sudo ./aws/install"
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    print_error "AWS credentials not configured. Please run: aws configure"
    exit 1
fi

print_status "✅ AWS CLI configured for region: $REGION"

# Create key pair if it doesn't exist
print_header "Setting Up SSH Key Pair"
if ! aws ec2 describe-key-pairs --key-names "$KEY_NAME" --region "$REGION" > /dev/null 2>&1; then
    print_status "Creating SSH key pair: $KEY_NAME"
    aws ec2 create-key-pair --key-name "$KEY_NAME" --region "$REGION" \
        --query 'KeyMaterial' --output text > "${KEY_NAME}.pem"
    chmod 400 "${KEY_NAME}.pem"
    print_status "✅ SSH key saved as ${KEY_NAME}.pem"
else
    print_status "✅ SSH key pair already exists"
fi

# Create security group
print_header "Setting Up Security Group"
if ! aws ec2 describe-security-groups --group-names "$SECURITY_GROUP_NAME" --region "$REGION" > /dev/null 2>&1; then
    print_status "Creating security group: $SECURITY_GROUP_NAME"
    
    SECURITY_GROUP_ID=$(aws ec2 create-security-group \
        --group-name "$SECURITY_GROUP_NAME" \
        --description "Ghosteam V5 Security Group" \
        --region "$REGION" \
        --query 'GroupId' --output text)
    
    # Add security group rules
    print_status "Adding security group rules..."
    
    # SSH access
    aws ec2 authorize-security-group-ingress \
        --group-id "$SECURITY_GROUP_ID" \
        --protocol tcp --port 22 --cidr 0.0.0.0/0 \
        --region "$REGION"
    
    # HTTP/HTTPS
    aws ec2 authorize-security-group-ingress \
        --group-id "$SECURITY_GROUP_ID" \
        --protocol tcp --port 80 --cidr 0.0.0.0/0 \
        --region "$REGION"
    
    aws ec2 authorize-security-group-ingress \
        --group-id "$SECURITY_GROUP_ID" \
        --protocol tcp --port 443 --cidr 0.0.0.0/0 \
        --region "$REGION"
    
    # Application ports
    for port in 5000 8080 3000 9090 6566; do
        aws ec2 authorize-security-group-ingress \
            --group-id "$SECURITY_GROUP_ID" \
            --protocol tcp --port $port --cidr 0.0.0.0/0 \
            --region "$REGION"
    done
    
    print_status "✅ Security group created with ID: $SECURITY_GROUP_ID"
else
    SECURITY_GROUP_ID=$(aws ec2 describe-security-groups \
        --group-names "$SECURITY_GROUP_NAME" \
        --region "$REGION" \
        --query 'SecurityGroups[0].GroupId' --output text)
    print_status "✅ Security group already exists: $SECURITY_GROUP_ID"
fi

# Get latest Ubuntu AMI
print_header "Finding Latest Ubuntu AMI"
AMI_ID=$(aws ec2 describe-images \
    --owners 099720109477 \
    --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-22.04-amd64-server-*" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
    --output text \
    --region "$REGION")

print_status "✅ Using Ubuntu AMI: $AMI_ID"

# Launch EC2 instance
print_header "Launching EC2 Instance"

# Check if instance already exists
EXISTING_INSTANCE=$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=$INSTANCE_NAME" "Name=instance-state-name,Values=running,pending" \
    --region "$REGION" \
    --query 'Reservations[0].Instances[0].InstanceId' --output text 2>/dev/null || echo "None")

if [ "$EXISTING_INSTANCE" != "None" ] && [ "$EXISTING_INSTANCE" != "null" ]; then
    print_status "✅ Instance already exists: $EXISTING_INSTANCE"
    INSTANCE_ID="$EXISTING_INSTANCE"
else
    print_status "Launching new EC2 instance..."
    
    # Create user data script for instance initialization
    cat > user-data.sh << 'EOF'
#!/bin/bash
set -e

# Update system
apt-get update
apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker ubuntu

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install K3s (lightweight Kubernetes)
curl -sfL https://get.k3s.io | sh -
mkdir -p /home/ubuntu/.kube
cp /etc/rancher/k3s/k3s.yaml /home/ubuntu/.kube/config
chown ubuntu:ubuntu /home/ubuntu/.kube/config

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install

# Install Python and pip
apt-get install -y python3 python3-pip python3-venv git

# Create deployment directory
mkdir -p /home/ubuntu/ghosteam-v5
chown ubuntu:ubuntu /home/ubuntu/ghosteam-v5

# Signal completion
touch /home/ubuntu/setup-complete
EOF

    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id "$AMI_ID" \
        --count 1 \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$SECURITY_GROUP_ID" \
        --user-data file://user-data.sh \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME}]" \
        --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":50,"VolumeType":"gp3"}}]' \
        --region "$REGION" \
        --query 'Instances[0].InstanceId' --output text)
    
    print_status "✅ Instance launched: $INSTANCE_ID"
fi

# Wait for instance to be running
print_status "Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

print_status "✅ Instance is running at: $PUBLIC_IP"

# Wait for SSH to be available
print_status "Waiting for SSH to be available..."
while ! nc -z "$PUBLIC_IP" 22; do
    sleep 5
done

print_status "✅ SSH is available"

# Wait for setup to complete
print_status "Waiting for instance setup to complete..."
while ! ssh -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no ubuntu@"$PUBLIC_IP" "test -f /home/ubuntu/setup-complete" 2>/dev/null; do
    echo -n "."
    sleep 10
done
echo ""

print_header "🎉 AWS EC2 Instance Ready!"

echo ""
echo "📊 Instance Details:"
echo "   Instance ID: $INSTANCE_ID"
echo "   Public IP:   $PUBLIC_IP"
echo "   SSH Key:     ${KEY_NAME}.pem"
echo "   Region:      $REGION"
echo ""

echo "🔗 SSH Access:"
echo "   ssh -i ${KEY_NAME}.pem ubuntu@$PUBLIC_IP"
echo ""

echo "📋 Next Steps:"
echo "1. Copy deployment files to instance:"
echo "   scp -i ${KEY_NAME}.pem -r ghosteam-v5/ ubuntu@$PUBLIC_IP:~/"
echo ""
echo "2. Deploy Ghosteam V5:"
echo "   ssh -i ${KEY_NAME}.pem ubuntu@$PUBLIC_IP"
echo "   cd ghosteam-v5"
echo "   ./scripts/deploy_cloud.sh"
echo ""

echo "🌐 Future Access Points (after deployment):"
echo "   API:      http://$PUBLIC_IP:8080"
echo "   MLflow:   http://$PUBLIC_IP:5000"
echo "   Grafana:  http://$PUBLIC_IP:3000"
echo ""

# Save connection info
cat > aws-connection-info.txt << EOF
# Ghosteam V5 AWS EC2 Connection Information
Instance ID: $INSTANCE_ID
Public IP: $PUBLIC_IP
SSH Command: ssh -i ${KEY_NAME}.pem ubuntu@$PUBLIC_IP
Region: $REGION

# Access URLs (after deployment):
API: http://$PUBLIC_IP:8080
MLflow: http://$PUBLIC_IP:5000
Grafana: http://$PUBLIC_IP:3000
Prometheus: http://$PUBLIC_IP:9090

# Deployment Commands:
scp -i ${KEY_NAME}.pem -r ghosteam-v5/ ubuntu@$PUBLIC_IP:~/
ssh -i ${KEY_NAME}.pem ubuntu@$PUBLIC_IP 'cd ghosteam-v5 && ./scripts/deploy_cloud.sh'
EOF

print_status "✅ Connection info saved to aws-connection-info.txt"
print_status "🚀 Ready to deploy Ghosteam V5!"
