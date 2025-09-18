# Ghosteam V5 Learning System

A comprehensive AI/ML learning system following modern MLOps best practices with microservices architecture, continuous learning capabilities, and production-grade infrastructure.

## 🚀 Features

- **Continuous Learning**: Automated drift detection and model retraining
- **Feature Store**: Real-time and batch feature serving with Feast
- **Model Serving**: High-performance inference with KServe and FastAPI
- **Monitoring**: Comprehensive observability with Prometheus and Grafana
- **MLOps Pipeline**: End-to-end ML workflows with Kubeflow Pipelines
- **Production Ready**: Kubernetes-native deployment with auto-scaling

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Feature Store  │    │  Model Training │
│                 │───▶│     (Feast)     │───▶│   (Kubeflow)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │    │  Model Serving  │    │  Model Registry │
│ (Prometheus)    │◀───│   (KServe)      │◀───│   (MLflow)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Development Setup

```bash
# Clone and setup environment
git clone <repository-url>
cd ghosteam-v5
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Start services
docker-compose up -d

# Initialize feature store
cd feast_repo && feast apply && cd ..

# Run tests
pytest tests/ -v

# Start API
uvicorn src.serving.api.main:app --host 0.0.0.0 --port 8080
```

### Production Deployment

```bash
# Setup Kubernetes environment
./scripts/setup_environment.sh

# Deploy to production
./scripts/deploy.sh
```

## 📊 Monitoring

- **MLflow UI**: http://localhost:5000
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **API Docs**: http://localhost:8080/docs

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/e2e/ -v

# Performance tests
pytest tests/performance/ -v
```

## 📚 Documentation

- [Architecture Overview](docs/architecture.md)
- [Deployment Guide](docs/deployment.md)
- [Model Cards](docs/model-cards/)
- [API Reference](http://localhost:8080/docs)

## 🔧 Configuration

Key configuration files:
- `src/config/settings.py` - Application settings
- `configs/model_configs/` - Model configurations
- `configs/pipeline_configs/` - Pipeline configurations
- `infrastructure/kubernetes/` - K8s deployments

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.
