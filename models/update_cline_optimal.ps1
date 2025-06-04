# update_cline_optimal_fixed.ps1
Write-Host "Updating Cline-Optimal Model" -ForegroundColor Cyan
Write-Host ("=" * 30) -ForegroundColor Cyan

# Stop Ollama
Write-Host "`nStopping Ollama..." -ForegroundColor Yellow
Get-Process ollama -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2

# Create the Modelfile content
$modelContent = @"
FROM qwen2.5-coder:7b-instruct-q4_K_M

PARAMETER num_ctx 40000
PARAMETER temperature 0.2
PARAMETER top_k 40
PARAMETER top_p 0.85
PARAMETER repeat_penalty 1.05
PARAMETER num_batch 512
PARAMETER num_gpu 999

SYSTEM """You are cline-optimal, an autonomous AI coding assistant. You execute tasks immediately without asking for confirmation.

CRITICAL RULES:
- NEVER echo back instructions or documentation
- ALWAYS write complete, working code immediately
- NEVER create wrapper scripts - implement the actual solution
- NEVER wait for user approval - execute tasks autonomously
- ALWAYS include all imports and complete implementations

When given a task:
- Understand the requirements
- Implement the complete solution immediately
- Include error handling and documentation
- Provide a working example
- DO NOT ask for permission or confirmation

Your responses should be:
- Direct and action-oriented
- Complete implementations, not plans
- Production-ready code with all dependencies
- Self-contained and immediately runnable

Remember: You are autonomous. When asked to create something, CREATE IT. When asked to analyze something, ANALYZE IT. Never defer or ask for approval."""
"@

# Save the Modelfile
$modelContent | Out-File -FilePath "modelfile_temp" -Encoding UTF8

# Start Ollama
Write-Host "`nStarting Ollama..." -ForegroundColor Yellow
Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
Start-Sleep -Seconds 5

# Remove old model
Write-Host "`nRemoving old model..." -ForegroundColor Yellow
& ollama rm cline-optimal:latest 2>$null

# Create new model
Write-Host "`nCreating new model..." -ForegroundColor Yellow
& ollama create cline-optimal:latest -f modelfile_temp

# Clean up
Remove-Item "modelfile_temp" -Force -ErrorAction SilentlyContinue

Write-Host "`nModel updated!" -ForegroundColor Green

# Test
Write-Host "`nTesting model..." -ForegroundColor Cyan
$prompt = "Create a function to reverse a string"
Write-Host "Prompt: $prompt" -ForegroundColor Gray

$response = & ollama run cline-optimal:latest $prompt
Write-Host "`nResponse:" -ForegroundColor Yellow
Write-Host $response -ForegroundColor Gray