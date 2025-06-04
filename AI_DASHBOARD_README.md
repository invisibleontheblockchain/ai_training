# AI Training Dashboard - RTX 3090 Optimized
## Real-time GPU Monitoring, Model Benchmarking & AI Prompt Generation

A comprehensive web-based dashboard for monitoring RTX 3090 performance, running model benchmarks, and generating AI prompts with advanced optimization features.

## 🚀 Features

### 🖥️ GPU Dashboard
- **Real-time RTX 3090 monitoring**
- GPU utilization, memory usage, temperature tracking
- Power consumption and clock speed monitoring
- Memory allocation visualization
- System performance metrics

### 🤖 Model Benchmark
- **Automated model performance testing**
- Support for multiple LLMs (Phi-2, Llama-2, Mistral, CodeLlama)
- RTX 3090 optimizations (Flash Attention, torch.compile)
- Configurable benchmark parameters
- Performance comparison visualizations

### 🎨 AI Prompt Generator
- **Smart prompt generation for image AI**
- Multiple art styles (Photorealistic, Cinematic, Fantasy, etc.)
- Subject categorization and customization
- Mood and atmosphere controls
- Advanced settings (aspect ratio, quality, seed)
- Prompt history tracking

### 📊 Performance Analysis
- **Historical performance tracking**
- WandB integration for training runs
- Performance trend visualization
- Optimization recommendations
- Comparative analysis tools

## 🛠️ Installation & Setup

### Prerequisites
- **Windows 10/11**
- **Python 3.8+**
- **NVIDIA RTX 3090 with latest drivers**
- **CUDA 11.8+ installed**

### Quick Start

1. **Clone or download the files to your AI_Training directory**

2. **Run the setup script with installation:**
   ```powershell
   .\launch_ai_dashboard.ps1 -Install
   ```

3. **Launch the dashboard:**
   ```powershell
   .\launch_ai_dashboard.ps1
   ```

4. **Open your browser to:**
   ```
   http://localhost:8501
   ```

### Manual Installation

If you prefer manual setup:

```powershell
# Install PyTorch with CUDA support
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dashboard requirements
pip install -r dashboard_requirements.txt

# Run the dashboard
streamlit run ai_dashboard.py
```

## 🎛️ Usage Guide

### GPU Dashboard Mode
- Monitor real-time RTX 3090 performance
- Track temperature, power, and utilization
- View memory allocation breakdown
- System resource monitoring

### Model Benchmark Mode
- Select models to benchmark
- Configure optimization settings:
  - ✅ Flash Attention 2
  - ✅ torch.compile optimization
  - ✅ System prompt injection
- Set benchmark parameters (tokens, temperature, runs)
- View performance comparisons

### Prompt Generator Mode
- Choose art style and subject category
- Add mood and custom details
- Generate optimized prompts for:
  - DALL-E, Midjourney, Stable Diffusion
  - Any image generation AI tool
- Save and review prompt history

### Performance Analysis Mode
- View historical performance trends
- Analyze WandB training data
- Get optimization recommendations
- Compare performance metrics

## ⚙️ Configuration

### RTX 3090 Optimizations
The dashboard automatically applies RTX 3090-specific optimizations:

```python
# Environment variables set automatically
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
TORCH_CUDA_ARCH_LIST=8.6
```

### Custom Settings
Modify settings in the sidebar:
- Auto-refresh interval
- Benchmark parameters
- Visualization preferences
- Performance thresholds

## 📁 File Structure

```
AI_Training/
├── ai_dashboard.py              # Main dashboard application
├── launch_ai_dashboard.ps1      # Setup and launch script
├── dashboard_requirements.txt   # Python dependencies
├── models/                      # AI models and utilities
│   ├── enhanced_model_benchmark.py
│   ├── system_prompt_injector.py
│   └── enhanced_evaluation.py
├── .streamlit/                  # Auto-generated config
│   └── config.toml
└── wandb/                       # Training run data
```

## 🔧 Troubleshooting

### Common Issues

**Dashboard won't start:**
```powershell
# Reinstall dependencies
.\launch_ai_dashboard.ps1 -Install -Update
```

**CUDA not detected:**
- Verify NVIDIA drivers are installed
- Check CUDA toolkit installation
- Restart after driver updates

**Memory errors:**
- Close other GPU applications
- Reduce batch sizes in benchmarks
- Check available VRAM

**Port conflicts:**
```powershell
# Use different port
.\launch_ai_dashboard.ps1 -Port 8502
```

### Performance Tips

1. **For best GPU monitoring:**
   - Install `nvidia-ml-py` for detailed stats
   - Enable auto-refresh for real-time data

2. **For model benchmarking:**
   - Use Flash Attention for 15-20% speed boost
   - Enable torch.compile for additional optimization
   - Adjust batch size based on model size

3. **For prompt generation:**
   - Use specific subjects for better results
   - Combine multiple styles for unique outputs
   - Save successful prompts for reuse

## 🚨 Safety & Monitoring

### Temperature Monitoring
- **Safe operating range:** <80°C
- **Optimal range:** 60-70°C
- **Auto-alerts:** Dashboard shows warnings if temps exceed safe limits

### Power Management
- **RTX 3090 TDP:** 350W maximum
- **Efficient range:** 200-300W under load
- **Monitoring:** Real-time power consumption tracking

### Memory Management
- **Total VRAM:** 24GB
- **Recommended usage:** <90% for stability
- **Auto-cleanup:** Memory cleared between benchmark runs

## 🔄 Updates & Maintenance

### Updating the Dashboard
```powershell
# Update dependencies
.\launch_ai_dashboard.ps1 -Update

# Or manually
pip install -r dashboard_requirements.txt --upgrade
```

### Performance Data
- Benchmark results saved to `enhanced_benchmark_results.json`
- WandB data automatically tracked in `wandb/` directory
- Performance logs maintained for historical analysis

## 🤝 Integration

### With Existing Workflows
- **WandB integration:** Automatic training run detection
- **Model compatibility:** Works with HuggingFace transformers
- **Benchmark results:** JSON export for external analysis

### API Endpoints
The dashboard runs on Streamlit, providing:
- Web interface on `http://localhost:8501`
- Real-time updates via WebSocket
- Configurable refresh intervals

## 📞 Support

### Getting Help
1. Check the troubleshooting section above
2. Verify all prerequisites are installed
3. Run with `-Install` flag to reset dependencies
4. Check GPU drivers and CUDA installation

### Performance Issues
- Monitor GPU temperature and usage
- Adjust benchmark parameters if needed
- Ensure sufficient system memory (32GB+ recommended)
- Close unnecessary applications during benchmarking

---

## 🎯 Quick Commands

```powershell
# First time setup
.\launch_ai_dashboard.ps1 -Install

# Normal launch
.\launch_ai_dashboard.ps1

# Update dependencies
.\launch_ai_dashboard.ps1 -Update

# Custom port
.\launch_ai_dashboard.ps1 -Port 8502

# Different host (for network access)
.\launch_ai_dashboard.ps1 -Host 0.0.0.0 -Port 8501
```

**Ready to optimize your RTX 3090 AI training workflow!** 🚀
