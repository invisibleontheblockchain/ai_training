# AI Training Dashboard Launcher
# PowerShell script to set up and launch the Streamlit frontend

param(
    [switch]$InstallDeps = $false,
    [string]$Port = "8501"
)

Write-Host @"
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    AI TRAINING DASHBOARD LAUNCHER                             ║
║                    RTX 3090 Performance & Image Prompts                       ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

# Check if Python is available
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "❌ Python not found. Please install Python 3.8+ first." -ForegroundColor Red
    exit 1
}

Write-Host "✓ Python found: $($pythonCmd.Source)" -ForegroundColor Green

# Install dependencies if requested
if ($InstallDeps) {
    Write-Host "`n📦 Installing dependencies..." -ForegroundColor Yellow
    
    try {
        pip install -r frontend_requirements.txt
        Write-Host "✅ Dependencies installed successfully" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Failed to install dependencies: $_" -ForegroundColor Red
        exit 1
    }
}

# Check if streamlit is available
try {
    $streamlitCheck = python -c "import streamlit; print('OK')" 2>$null
    if ($streamlitCheck -ne "OK") {
        throw "Streamlit not available"
    }
}
catch {
    Write-Host "❌ Streamlit not found. Installing..." -ForegroundColor Yellow
    pip install streamlit
}

# Set environment variables for optimal performance
$env:STREAMLIT_SERVER_PORT = $Port
$env:STREAMLIT_SERVER_ADDRESS = "0.0.0.0"
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"

# Launch the application
Write-Host "`n🚀 Launching AI Training Dashboard..." -ForegroundColor Green
Write-Host "   📍 URL: http://localhost:$Port" -ForegroundColor Cyan
Write-Host "   🔧 Use Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

try {
    streamlit run frontend_app.py --server.port $Port --server.address 0.0.0.0
}
catch {
    Write-Host "❌ Failed to launch application: $_" -ForegroundColor Red
    Write-Host "`n💡 Try running with -InstallDeps flag to install missing dependencies" -ForegroundColor Yellow
}
