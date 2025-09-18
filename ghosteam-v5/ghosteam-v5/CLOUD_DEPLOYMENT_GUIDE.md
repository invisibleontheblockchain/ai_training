# 🚀 **GHOSTEAM V5 CLOUD DEPLOYMENT GUIDE**

Deploy your complete MLOps system to the cloud in minutes! This guide provides step-by-step instructions for deploying Ghosteam V5 to AWS, Google Cloud, or Azure.

## 🎯 **What You'll Get**

A complete cloud-hosted MLOps system including:
- ✅ **MLflow Tracking Server** - Experiment tracking and model registry
- ✅ **Feature Store (Feast)** - Online and offline feature serving
- ✅ **Model Serving API** - FastAPI with auto-scaling
- ✅ **Monitoring Stack** - Prometheus and Grafana dashboards
- ✅ **Databases** - PostgreSQL and Redis
- ✅ **Sample Data & Models** - Ready-to-use examples

## 🌩️ **OPTION 1: AWS EC2 (RECOMMENDED)**

### **Prerequisites:**
- AWS CLI installed and configured
- AWS account with EC2 permissions

### **Quick Deploy:**
```bash
cd ghosteam-v5
./scripts/deploy_aws_ec2.sh
```

### **What it does:**
1. ✅ Creates EC2 instance (t3.xlarge, 4 vCPU, 16GB RAM)
2. ✅ Sets up security groups for all required ports
3. ✅ Installs Docker, K3s, and all dependencies
4. ✅ Configures 50GB storage
5. ✅ Provides SSH key for access

### **Manual Steps:**
If you prefer manual setup:

```bash
# 1. Launch EC2 instance
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --count 1 \
  --instance-type t3.xlarge \
  --key-name your-key-name \
  --security-group-ids sg-xxxxxxxxx

# 2. SSH to instance
ssh -i your-key.pem ubuntu@YOUR_PUBLIC_IP

# 3. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# 4. Copy and deploy
scp -i your-key.pem -r ghosteam-v5/ ubuntu@YOUR_PUBLIC_IP:~/
ssh -i your-key.pem ubuntu@YOUR_PUBLIC_IP 'cd ghosteam-v5 && ./scripts/deploy_cloud.sh'
```

## ☁️ **OPTION 2: Google Cloud Platform**

### **Prerequisites:**
- Google Cloud CLI installed
- GCP project with Compute Engine enabled

### **Quick Deploy:**
```bash
cd ghosteam-v5
export GCP_PROJECT_ID=your-project-id
./scripts/deploy_gcp.sh
```

### **Manual Steps:**
```bash
# 1. Create instance
gcloud compute instances create ghosteam-v5-instance \
  --zone=us-central1-a \
  --machine-type=e2-standard-4 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB

# 2. Create firewall rule
gcloud compute firewall-rules create ghosteam-v5-ports \
  --allow tcp:22,tcp:80,tcp:443,tcp:3000,tcp:5000,tcp:8080,tcp:9090

# 3. SSH and deploy
gcloud compute ssh ghosteam-v5-instance --zone=us-central1-a
# Then follow Docker installation and deployment steps
```

## 🔷 **OPTION 3: Microsoft Azure**

### **Prerequisites:**
- Azure CLI installed and logged in
- Azure subscription

### **Quick Deploy:**
```bash
cd ghosteam-v5
./scripts/deploy_azure.sh  # Coming soon
```

### **Manual Steps:**
```bash
# 1. Create resource group
az group create --name ghosteam-v5-rg --location eastus

# 2. Create VM
az vm create \
  --resource-group ghosteam-v5-rg \
  --name ghosteam-v5-vm \
  --image Ubuntu2204 \
  --size Standard_D4s_v3 \
  --admin-username azureuser \
  --generate-ssh-keys

# 3. Open ports
az vm open-port --resource-group ghosteam-v5-rg --name ghosteam-v5-vm --port 22,80,443,3000,5000,8080,9090

# 4. SSH and deploy
ssh azureuser@YOUR_PUBLIC_IP
# Then follow Docker installation and deployment steps
```

## 🚀 **UNIVERSAL CLOUD DEPLOYMENT**

Once you have a cloud instance with Docker installed, use this universal deployment script:

```bash
# Copy files to your cloud instance
scp -r ghosteam-v5/ user@YOUR_PUBLIC_IP:~/

# SSH to instance
ssh user@YOUR_PUBLIC_IP

# Deploy the system
cd ghosteam-v5
./scripts/deploy_cloud.sh
```

## 📊 **Access Your Deployed System**

After deployment completes (5-10 minutes), access your MLOps system:

```
🚀 API:              http://YOUR_PUBLIC_IP:8080
📓 API Docs:         http://YOUR_PUBLIC_IP:8080/docs
📊 MLflow:           http://YOUR_PUBLIC_IP:5000
📈 Grafana:          http://YOUR_PUBLIC_IP:3000 (admin/admin123)
📉 Prometheus:       http://YOUR_PUBLIC_IP:9090
🍃 Feast:            http://YOUR_PUBLIC_IP:6566
```

## 🧪 **Test Your Deployment**

```bash
# Test API health
curl http://YOUR_PUBLIC_IP:8080/health

# Test model prediction
curl -X POST http://YOUR_PUBLIC_IP:8080/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "data": [{"age": 25, "activity_score": 0.8}],
    "model_name": "initial_model"
  }'

# Check MLflow models
curl http://YOUR_PUBLIC_IP:5000/api/2.0/mlflow/registered-models/search
```

## 🔧 **System Management**

### **Monitor Services:**
```bash
# Check all services
docker-compose -f docker-compose.cloud.yml ps

# View logs
docker-compose -f docker-compose.cloud.yml logs -f api

# Restart services
docker-compose -f docker-compose.cloud.yml restart
```

### **Scale Services:**
```bash
# Scale API service
docker-compose -f docker-compose.cloud.yml up -d --scale api=3

# Update configuration
docker-compose -f docker-compose.cloud.yml down
docker-compose -f docker-compose.cloud.yml up -d
```

## 🛡️ **Security Considerations**

### **Production Security:**
1. **Change default passwords** in docker-compose.cloud.yml
2. **Set up SSL/TLS** with Let's Encrypt or cloud provider certificates
3. **Configure firewall** to restrict access to specific IPs
4. **Enable authentication** on Grafana and other services
5. **Use secrets management** for sensitive configuration

### **Quick Security Setup:**
```bash
# Change Grafana password
docker-compose -f docker-compose.cloud.yml exec grafana grafana-cli admin reset-admin-password newpassword

# Restrict access (example for AWS)
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxxxxxx \
  --protocol tcp --port 3000 \
  --source-group sg-xxxxxxxxx  # Only allow access from specific security group
```

## 💰 **Cost Optimization**

### **Instance Sizing:**
- **Development**: t3.large (2 vCPU, 8GB RAM) - ~$60/month
- **Production**: t3.xlarge (4 vCPU, 16GB RAM) - ~$120/month
- **High Load**: t3.2xlarge (8 vCPU, 32GB RAM) - ~$240/month

### **Cost Saving Tips:**
1. **Use spot instances** for development (50-90% savings)
2. **Stop instances** when not in use
3. **Use reserved instances** for production (up to 75% savings)
4. **Monitor usage** with cloud provider cost tools

## 🔄 **Backup & Recovery**

### **Data Backup:**
```bash
# Backup volumes
docker run --rm -v ghosteam-v5_postgres_data:/data -v $(pwd):/backup ubuntu tar czf /backup/postgres_backup.tar.gz /data

# Backup MLflow artifacts
docker run --rm -v ghosteam-v5_mlflow_data:/data -v $(pwd):/backup ubuntu tar czf /backup/mlflow_backup.tar.gz /data
```

### **System Snapshot:**
```bash
# AWS - Create AMI
aws ec2 create-image --instance-id i-xxxxxxxxx --name "ghosteam-v5-backup-$(date +%Y%m%d)"

# GCP - Create snapshot
gcloud compute disks snapshot ghosteam-v5-instance --zone=us-central1-a
```

## 🚨 **Troubleshooting**

### **Common Issues:**

**Services not starting:**
```bash
# Check logs
docker-compose -f docker-compose.cloud.yml logs

# Restart with fresh containers
docker-compose -f docker-compose.cloud.yml down -v
docker-compose -f docker-compose.cloud.yml up -d
```

**Out of memory:**
```bash
# Check memory usage
docker stats

# Increase swap space
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**Port conflicts:**
```bash
# Check what's using ports
sudo netstat -tulpn | grep :8080

# Kill conflicting processes
sudo fuser -k 8080/tcp
```

## 📞 **Support**

If you encounter issues:

1. **Check logs**: `docker-compose -f docker-compose.cloud.yml logs`
2. **Verify ports**: `netstat -tulpn | grep -E ':(3000|5000|8080|9090)'`
3. **Check disk space**: `df -h`
4. **Monitor resources**: `htop` or `docker stats`

## 🎉 **Success!**

You now have a complete MLOps system running in the cloud! Your Ghosteam V5 deployment includes:

✅ **Model Training & Registry** - MLflow for experiment tracking  
✅ **Feature Store** - Feast for feature management  
✅ **Model Serving** - FastAPI with auto-scaling  
✅ **Monitoring** - Prometheus & Grafana dashboards  
✅ **Data Storage** - PostgreSQL & Redis  
✅ **Sample Data** - Ready-to-use examples  

**Ready to build amazing ML applications!** 🚀
