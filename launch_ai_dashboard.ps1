# AI Training Dashboard Launch Script
# RTX 3090 Optimized Setup and Launch

param(
    [switch]$Install = $false,
    [switch]$Update = $false,
    [string]$Port = "8501",
    [string]$Host = "localhost"
)

Write-Host @"
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    AI TRAINING DASHBOARD - RTX 3090                           ║
║                    GPU Performance & Model Benchmarking                       ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

# Set optimal environment for RTX 3090
Write-Host "`n🔧 Setting RTX 3090 environment variables..." -ForegroundColor Yellow

$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:512"
$env:TORCH_CUDA_ARCH_LIST = "8.6"  # RTX 3090 architecture
$env:CUDA_LAUNCH_BLOCKING = "0"
$env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"

# Function to check if command exists
function Test-Command {
    param($Command)
    try {
        Get-Command $Command -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

# Check Python installation
if (-not (Test-Command "python")) {
    Write-Host "❌ Python not found. Please install Python 3.8+ first." -ForegroundColor Red
    exit 1
}

$pythonVersion = python --version 2>&1
Write-Host "✅ Found: $pythonVersion" -ForegroundColor Green

# Check CUDA installation
if (Test-Command "nvidia-smi") {
    try {
        $gpuInfo = nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
        Write-Host "✅ NVIDIA GPU detected: $gpuInfo" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  NVIDIA drivers detected but GPU query failed" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠️  NVIDIA drivers not detected. GPU acceleration may not be available." -ForegroundColor Yellow
}

# Install or update dependencies
if ($Install -or $Update) {
    Write-Host "`n📦 Installing/updating dependencies..." -ForegroundColor Yellow
    
    # Upgrade pip first
    python -m pip install --upgrade pip
    
    # Install PyTorch with CUDA support for RTX 3090
    Write-Host "Installing PyTorch with CUDA 11.8 support..."
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    # Install other requirements
    if (Test-Path "dashboard_requirements.txt") {
        python -m pip install -r dashboard_requirements.txt
    } else {
        Write-Host "⚠️  dashboard_requirements.txt not found. Installing basic requirements..."
        python -m pip install streamlit plotly pandas numpy psutil transformers accelerate
    }
    
    Write-Host "✅ Dependencies installed!" -ForegroundColor Green
}

# Verify PyTorch CUDA installation
Write-Host "`n🔍 Verifying PyTorch CUDA installation..." -ForegroundColor Yellow
$torchTest = python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>&1

if ($torchTest -match "CUDA Available: True") {
    Write-Host "✅ PyTorch CUDA installation verified!" -ForegroundColor Green
    Write-Host $torchTest -ForegroundColor Cyan
} else {
    Write-Host "⚠️  PyTorch CUDA not available. Check installation." -ForegroundColor Yellow
    Write-Host $torchTest -ForegroundColor Red
}

# Check if main dashboard file exists
if (-not (Test-Path "ai_dashboard.py")) {
    Write-Host "❌ ai_dashboard.py not found in current directory." -ForegroundColor Red
    Write-Host "Please ensure you're in the correct directory with the dashboard files." -ForegroundColor Yellow
    exit 1
}

# Create launch configuration
$streamlitConfig = @"
[global]
developmentMode = false
showWarningOnDirectExecution = false

[server]
headless = false
enableCORS = false
enableXsrfProtection = false
maxUploadSize = 1028

[browser]
gatherUsageStats = false

[theme]
base = "dark"
primaryColor = "#00ff88"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#262730"
textColor = "#fafafa"
"@

# Create .streamlit directory and config
if (-not (Test-Path ".streamlit")) {
    New-Item -ItemType Directory -Path ".streamlit" -Force | Out-Null
}
$streamlitConfig | Out-File -FilePath ".streamlit/config.toml" -Encoding UTF8

Write-Host "`n🚀 Launching AI Training Dashboard..." -ForegroundColor Green
Write-Host "   URL: http://$Host`:$Port" -ForegroundColor Cyan
Write-Host "   Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host "`n" + "="*80 -ForegroundColor Cyan

# Launch Streamlit
try {
    streamlit run ai_dashboard.py --server.port $Port --server.address $Host
} catch {
    Write-Host "`n❌ Error launching dashboard: $_" -ForegroundColor Red
    Write-Host "Try running with -Install flag to install dependencies." -ForegroundColor Yellow
}

Write-Host "`n👋 Dashboard stopped. Thank you for using AI Training Dashboard!" -ForegroundColor Green
