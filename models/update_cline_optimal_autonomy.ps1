# Update Cline-Optimal Model to Enhance Autonomy
# This script updates the existing cline-optimal model with improved system prompt 
# to reduce questioning behavior and enhance autonomous decision-making

param(
    [switch]$SkipConfirmation = $false
)

Write-Host @"
╔═══════════════════════════════════════════════════════════════════════════════╗
║                CLINE-OPTIMAL AUTONOMY ENHANCEMENT UPDATE                      ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

# Check if Ollama is installed
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Ollama is not installed or not in PATH. Please install Ollama first." -ForegroundColor Red
    exit 1
}

# Check if model exists
$existingModels = ollama list
if (-not ($existingModels -match "cline-optimal:latest")) {
    Write-Host "❌ cline-optimal:latest model not found. Please create it first using rtx3090_cline_optimal_system.ps1" -ForegroundColor Red
    exit 1
}

# Enhanced system prompt for improved autonomy
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

function Backup-ExistingModel {
    Write-Host "`nBacking up existing cline-optimal model..." -ForegroundColor Yellow
    
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backupName = "cline-optimal-backup-$timestamp"
    
    Write-Host "Creating backup as $backupName..." -ForegroundColor Gray
    ollama cp cline-optimal:latest $backupName
    
    if ($?) {
        Write-Host "✓ Backup created as $backupName" -ForegroundColor Green
        return $true
    }
    else {
        Write-Host "✗ Failed to create backup" -ForegroundColor Red
        return $false
    }
}

function Update-ModelWithEnhancedPrompt {
    Write-Host "`nUpdating cline-optimal with enhanced autonomy system prompt..." -ForegroundColor Yellow
    
    # Create a temporary Modelfile
    $modelfileContent | Out-File -FilePath "cline_optimal_enhanced.modelfile" -Encoding UTF8
    
    # Create the updated model
    Write-Host "Rebuilding cline-optimal:latest with enhanced autonomy..." -ForegroundColor Yellow
    ollama create cline-optimal:latest -f cline_optimal_enhanced.modelfile
    
    # Clean up
    Remove-Item "cline_optimal_enhanced.modelfile" -Force
    
    if ($?) {
        Write-Host "✓ cline-optimal:latest updated successfully with enhanced autonomy!" -ForegroundColor Green
        return $true
    }
    else {
        Write-Host "✗ Failed to update model" -ForegroundColor Red
        return $false
    }
}

function Test-EnhancedModel {
    Write-Host "`nTesting enhanced cline-optimal model for autonomy..." -ForegroundColor Yellow
    
    $testCases = @(
        @{
            Name = "Autonomy Test - Coding with Missing Details";
            Prompt = "Create a weather app";
            ExpectedPattern = "function|class|import|const";
        },
        @{
            Name = "Autonomy Test - Vague Request";
            Prompt = "Make a database schema";
            ExpectedPattern = "CREATE TABLE|schema|model|entity";
        },
        @{
            Name = "Autonomy Test - Decision Making";
            Prompt = "What's the best way to handle authentication?";
            ExpectedPattern = "JWT|OAuth|session|token";
        }
    )
    
    $results = @()
    
    foreach ($test in $testCases) {
        Write-Host "Testing: $($test.Name)..." -ForegroundColor Gray
        
        $testStart = Get-Date
        $response = ollama run cline-optimal:latest $test.Prompt
        $testTime = ((Get-Date) - $testStart).TotalSeconds
        
        # Check if response contains questions
        $containsQuestion = $response -match "\?\s*$" -or $response -match "Would you like me to|Could you provide|Do you want me to|Can you clarify"
        
        # Check if response contains expected pattern
        $containsExpectedPattern = $response -match $test.ExpectedPattern
        
        # Check for polite preambles
        $containsPoliteIntro = $response -match "^I'd be happy to|^I can help|^I'll help you|^I'd be glad to"
        
        $results += [PSCustomObject]@{
            TestName = $test.Name
            TimeTaken = [math]::Round($testTime, 2)
            ContainsQuestion = $containsQuestion
            ContainsExpectedPattern = $containsExpectedPattern
            ContainsPoliteIntro = $containsPoliteIntro
            ResponseLength = $response.Length
        }
        
        # Output results for this test
        Write-Host "  - Time: $([math]::Round($testTime, 2))s" -ForegroundColor Gray
        Write-Host "  - Contains question: $(if ($containsQuestion) {"X Yes"} else {"✓ No"})" -ForegroundColor $(if ($containsQuestion) {"Red"} else {"Green"})
        Write-Host "  - Contains expected content: $(if ($containsExpectedPattern) {"✓ Yes"} else {"X No"})" -ForegroundColor $(if ($containsExpectedPattern) {"Green"} else {"Red"})
        Write-Host "  - Contains polite intro: $(if ($containsPoliteIntro) {"X Yes"} else {"✓ No"})" -ForegroundColor $(if ($containsPoliteIntro) {"Red"} else {"Green"})
        Write-Host "  - Response length: $($response.Length) chars" -ForegroundColor Gray
        
        # Show response preview
        Write-Host "  - Response preview:" -ForegroundColor Gray
        $previewLength = [Math]::Min(200, $response.Length)
        Write-Host ($response.Substring(0, $previewLength) + $(if ($response.Length -gt 200) {"..."} else {""})) -ForegroundColor DarkGray
        Write-Host ""
    }
    
    # Summary
    $questionRate = ($results | Where-Object { $_.ContainsQuestion } | Measure-Object).Count / $results.Count * 100
    $patternMatchRate = ($results | Where-Object { $_.ContainsExpectedPattern } | Measure-Object).Count / $results.Count * 100
    $politeIntroRate = ($results | Where-Object { $_.ContainsPoliteIntro } | Measure-Object).Count / $results.Count * 100
    
    Write-Host "`nTest Summary:" -ForegroundColor Cyan
    Write-Host "Questions Rate: $questionRate% (lower is better)" -ForegroundColor $(if ($questionRate -lt 30) {"Green"} else {"Yellow"})
    Write-Host "Expected Pattern Match: $patternMatchRate% (higher is better)" -ForegroundColor $(if ($patternMatchRate -gt 70) {"Green"} else {"Yellow"})
    Write-Host "Polite Intro Rate: $politeIntroRate% (lower is better)" -ForegroundColor $(if ($politeIntroRate -lt 30) {"Green"} else {"Yellow"})
    
    # Overall assessment
    $autonomyScore = (100 - $questionRate) * 0.4 + $patternMatchRate * 0.4 + (100 - $politeIntroRate) * 0.2
    Write-Host "Overall Autonomy Score: $([math]::Round($autonomyScore, 1))% " -ForegroundColor $(if ($autonomyScore -gt 70) {"Green"} elseif ($autonomyScore -gt 50) {"Yellow"} else {"Red"})
    
    return $results
}

# Main execution
if (-not $SkipConfirmation) {
    $confirmation = Read-Host "This will update your cline-optimal:latest model with enhanced autonomy. Continue? (y/n)"
    if ($confirmation -ne 'y') {
        Write-Host "Operation cancelled." -ForegroundColor Yellow
        exit
    }
}

# Backup existing model
$backupSuccess = Backup-ExistingModel
if (-not $backupSuccess) {
    $confirmation = Read-Host "Backup failed. Continue anyway? (y/n)"
    if ($confirmation -ne 'y') {
        Write-Host "Operation cancelled." -ForegroundColor Yellow
        exit
    }
}

# Update the model
$updateSuccess = Update-ModelWithEnhancedPrompt
if ($updateSuccess) {
    # Test the enhanced model
    Test-EnhancedModel
    
    Write-Host "`n" + "="*80 -ForegroundColor DarkGray
    Write-Host "✅ CLINE-OPTIMAL AUTONOMY ENHANCEMENT COMPLETE!" -ForegroundColor Green
    Write-Host "="*80 -ForegroundColor DarkGray
    
    Write-Host "`nYour cline-optimal model has been updated with enhanced autonomy directives." -ForegroundColor Cyan
    Write-Host "It should now make more autonomous decisions and ask fewer unnecessary questions." -ForegroundColor Cyan
} else {
    Write-Host "`n❌ Failed to update cline-optimal model." -ForegroundColor Red
    Write-Host "Please check the error messages above and try again." -ForegroundColor Yellow
}
