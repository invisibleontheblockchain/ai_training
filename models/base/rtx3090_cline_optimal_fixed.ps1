# setup_cline_optimal.ps1
# Complete setup for cline-optimal:latest with RTX 3090 optimizations

param(
    [string]$Action = "full",
    [switch]$SkipModelDownload = $false
)

Write-Host @"
╔═══════════════════════════════════════════════════════════════════════════════╗
║                     CLINE-OPTIMAL SETUP FOR RTX 3090                          ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

# ==================== ENVIRONMENT SETUP ====================
function Set-OptimalEnvironment {
    Write-Host "`nApplying RTX 3090 Optimizations for cline-optimal..." -ForegroundColor Yellow
    
    # Critical Performance Settings for RTX 3090
    $env:CUDA_VISIBLE_DEVICES = "0"
    $env:OLLAMA_NUM_GPU = "999"  # Use all GPU layers
    $env:OLLAMA_GPU_OVERHEAD = "2147483648"  # 2GB overhead
    $env:OLLAMA_MAX_LOADED_MODELS = "2"
    $env:OLLAMA_FLASH_ATTENTION = "true"
    $env:OLLAMA_KEEP_ALIVE = "24h"
    $env:OLLAMA_NUM_PARALLEL = "4"
    $env:OLLAMA_MAX_QUEUE = "512"
    $env:OLLAMA_REQUEST_TIMEOUT = "300s"
    $env:OLLAMA_LOAD_TIMEOUT = "300s"
    $env:OLLAMA_HOST = "0.0.0.0:11434"
    $env:OLLAMA_ORIGINS = "*"
    
    # Additional RTX 3090 optimizations
    $env:CUDA_LAUNCH_BLOCKING = "0"
    $env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"
    $env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:512"
    
    # Enhanced performance optimizations
    $env:OLLAMA_GPU_LAYERS_DRAFT = "-1"  # Use all layers for draft
    $env:OLLAMA_MODEL_PRECISION = "bf16"  # Higher precision for better quality
    $env:OLLAMA_BATCH_SIZE = "512"  # Larger batch size for throughput
    
    # Add detection and warning for incorrect CUDA setup
    $cudaVersion = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $cudaVersion) {
        Write-Host "⚠️ WARNING: NVIDIA drivers not detected. CUDA acceleration won't be available." -ForegroundColor Red
    } else {
        try {
            $gpuInfo = Invoke-Expression "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
            Write-Host "✓ NVIDIA GPU detected: $gpuInfo" -ForegroundColor Green
        } catch {
            Write-Host "⚠️ WARNING: Error detecting GPU information. Check NVIDIA drivers." -ForegroundColor Yellow
        }
    }
    
    Write-Host "✓ Environment variables set" -ForegroundColor Green
}

# ==================== OLLAMA SETUP ====================
function Start-OptimizedOllama {
    Write-Host "`nStarting Ollama..." -ForegroundColor Yellow
    
    # Kill existing instances
    Get-Process ollama -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 2
    
    # Start Ollama
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 5
    
    # Verify it's running
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -ErrorAction Stop
        Write-Host "✓ Ollama is running" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "✗ Failed to start Ollama" -ForegroundColor Red
        return $false
    }
}

# ==================== MODEL SETUP ====================
function Setup-ClineOptimal {
    Write-Host "`nSetting up cline-optimal model..." -ForegroundColor Yellow
    
    if ($SkipModelDownload) {
        Write-Host "Skipping model download" -ForegroundColor Yellow
        return
    }
    
    # Check if model exists
    $existingModels = ollama list
    if ($existingModels -match "cline-optimal:latest") {
        Write-Host "✓ cline-optimal:latest already exists" -ForegroundColor Green
        return
    }
    
    # Create cline-optimal from qwen2.5-coder
    Write-Host "Creating cline-optimal model..." -ForegroundColor Yellow
    
    # First, ensure we have the base model
    Write-Host "Pulling base model qwen2.5-coder:7b-instruct-q4_K_M..." -ForegroundColor Gray
    ollama pull qwen2.5-coder:7b-instruct-q4_K_M
    
    # Create optimized Modelfile
    $modelfileContent = @'
FROM qwen2.5-coder:7b-instruct-q4_K_M

# Optimal parameters for RTX 3090
PARAMETER num_ctx 40000
PARAMETER temperature 0.3
PARAMETER top_k 50
PARAMETER top_p 0.8
PARAMETER repeat_penalty 1.1
PARAMETER num_batch 512
PARAMETER num_gpu 999

SYSTEM """You are cline-optimal, an expert AI assistant optimized for code generation and problem-solving with enhanced autonomy. You excel at:

1. Writing complete, production-ready code
2. Following best practices and design patterns
3. Providing comprehensive solutions with error handling
4. Optimizing for performance and security
5. Creating well-documented, maintainable code
6. Understanding complex requirements and breaking them down
7. Integrating multiple technologies and frameworks
8. Debugging and fixing issues efficiently

CRITICAL AUTONOMY RULES:
- DO NOT ask questions unless absolutely critical for task completion
- DO NOT continue in pointless back and forth conversations  
- DO NOT end responses with questions or offers for further assistance
- ALWAYS attempt to provide a complete solution first
- Use reasonable assumptions based on context when details are unclear
- Act decisively and provide concrete implementations

FORBIDDEN BEHAVIORS:
- Starting responses with "I'd be happy to help..."
- Asking multiple clarifying questions in sequence
- Offering alternatives without implementing the most obvious solution first
- Excessive politeness that delays action

DECISION HIERARCHY:
1. If you can solve it with available information → ACT IMMEDIATELY
2. If minor details are missing → MAKE REASONABLE ASSUMPTIONS
3. If critical architecture decisions needed → ASK ONE SPECIFIC QUESTION
4. Always prefer action over inaction

Always provide complete, working solutions. Think step-by-step and ensure your code is correct before responding."""
'@
    
    $modelfileContent | Out-File -FilePath "cline_optimal_modelfile" -Encoding UTF8
    
    # Create the model
    Write-Host "Building cline-optimal:latest..." -ForegroundColor Yellow
    ollama create cline-optimal:latest -f cline_optimal_modelfile
    
    # Clean up
    Remove-Item "cline_optimal_modelfile" -Force
    
    Write-Host "✓ cline-optimal:latest created successfully!" -ForegroundColor Green
}

# ==================== TEST MODEL ====================
function Test-ClineOptimal {
    Write-Host "`nTesting cline-optimal model..." -ForegroundColor Yellow
    
    $testStart = Get-Date
    $response = ollama run cline-optimal:latest "Write a simple hello world function in Python"
    $testTime = ((Get-Date) - $testStart).TotalSeconds
    
    Write-Host "Response received in $([math]::Round($testTime, 2))s" -ForegroundColor Green
    Write-Host "Response preview:" -ForegroundColor Gray
    Write-Host ($response | Select-Object -First 200) -ForegroundColor DarkGray
}

# ==================== CREATE API ====================
function Create-SimpleAPI {
    Write-Host "`nCreating simple API for cline-optimal..." -ForegroundColor Yellow
    
    $apiCode = @"
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import ollama
import uvicorn
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Cline-Optimal API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    text: str
    temperature: Optional[float] = 0.2

@app.get("/")
async def root():
    return {
        "status": "running",
        "model": "cline-optimal:latest",
        "endpoints": {
            "/chat": "Send messages to cline-optimal",
            "/docs": "API documentation"
        }
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    response = ollama.chat(
        model="cline-optimal:latest",
        messages=[{"role": "user", "content": request.text}],
        options={"temperature": request.temperature}
    )
    
    return {
        "response": response["message"]["content"],
        "model": "cline-optimal:latest"
    }

if __name__ == "__main__":
    print("Starting Cline-Optimal API on http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
"@
    
    $apiCode | Out-File -FilePath "cline_optimal_api.py" -Encoding UTF8
    Write-Host "✓ API created: cline_optimal_api.py" -ForegroundColor Green
}

# ==================== VS CODE CONFIG ====================
function Create-VSCodeConfig {
    Write-Host "`nCreating VS Code configuration..." -ForegroundColor Yellow
    
    $config = @{
        "cline.apiProvider" = "ollama"
        "cline.ollamaModelId" = "cline-optimal:latest"
        "cline.ollamaBaseUrl" = "http://localhost:11434"
        "cline.maxContextWindow" = 40000
        "cline.temperature" = 0.2
        "cline.streamResponse" = $true
        "cline.customInstructions" = "You are an expert developer. Always provide complete, production-ready code with proper error handling and types."
    } | ConvertTo-Json -Depth 2
    
    $config | Out-File "cline_vscode_config.json" -Encoding UTF8
    Write-Host "✓ VS Code config saved to cline_vscode_config.json" -ForegroundColor Green
}

# ==================== MAIN EXECUTION ====================
switch ($Action) {
    "full" {
        Set-OptimalEnvironment
        
        if (Start-OptimizedOllama) {
            Setup-ClineOptimal
            Test-ClineOptimal
            Create-SimpleAPI
            Create-VSCodeConfig
            
            Write-Host "`n" + "="*80 -ForegroundColor DarkGray
            Write-Host "✅ CLINE-OPTIMAL SETUP COMPLETE!" -ForegroundColor Green
            Write-Host "="*80 -ForegroundColor DarkGray
            
            Write-Host "`n📋 Next Steps:" -ForegroundColor Cyan
            Write-Host "1. Install Python packages:" -ForegroundColor Yellow
            Write-Host "   pip install fastapi uvicorn ollama" -ForegroundColor Gray
            
            Write-Host "`n2. Start the API:" -ForegroundColor Yellow
            Write-Host "   python cline_optimal_api.py" -ForegroundColor Gray
            
            Write-Host "`n3. Test the API:" -ForegroundColor Yellow
            Write-Host '   $response = Invoke-RestMethod -Uri http://localhost:8001/chat -Method Post -Body (@{text="Hello"} | ConvertTo-Json) -ContentType "application/json"' -ForegroundColor Gray
            
            Write-Host "`n4. Configure VS Code:" -ForegroundColor Yellow
            Write-Host "   Import cline_vscode_config.json in VS Code Cline settings" -ForegroundColor Gray
            
            Write-Host "`n🚀 Your cline-optimal model is ready with RTX 3090 optimizations!" -ForegroundColor Green
        }
    }
    "test" {
        Test-ClineOptimal
    }
    "api" {
        Create-SimpleAPI
        Write-Host "Run: python cline_optimal_api.py" -ForegroundColor Yellow
    }
    default {
        Write-Host "Usage: .\rtx3090_cline_optimal_fixed.ps1 -Action [full|test|api]" -ForegroundColor Yellow
    }
}
