#!/usr/bin/env python3
"""
Advanced Coding Assistant
========================
Complete local Claude/Codex-type coding assistant that integrates all components
for research-driven, self-improving code generation and problem solving.
"""

import asyncio
import json
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import threading
import queue
import subprocess
from pathlib import Path

# Import our advanced components
from research_prompt_optimizer import ResearchPromptOptimizer, ResearchPromptConfig
from multi_model_orchestrator import ModelOrchestrator, ModelType
from self_improving_research_system import SelfImprovingResearchSystem
from master_agi_system import MasterAGISystem

@dataclass
class CodingTask:
    """Represents a coding task with context and requirements"""
    task_id: str
    description: str
    language: Optional[str]
    framework: Optional[str]
    complexity: int  # 1-5
    requirements: List[str]
    constraints: List[str]
    context: Dict[str, Any]
    priority: str  # low, medium, high, critical
    deadline: Optional[datetime]

@dataclass
class CodingResult:
    """Result from a coding session"""
    task_id: str
    code: str
    explanation: str
    tests: str
    documentation: str
    security_notes: str
    performance_notes: str
    deployment_guide: str
    quality_score: float
    execution_time: float
    model_used: str
    success: bool
    timestamp: datetime

class AdvancedCodingAssistant:
    """Complete coding assistant with research, multi-model, and self-improvement capabilities"""
    
    def __init__(self):
        self.setup_logging()
        
        # Initialize all components
        self.research_system = SelfImprovingResearchSystem()
        self.model_orchestrator = ModelOrchestrator()
        self.prompt_optimizer = ResearchPromptOptimizer()
        self.agi_system = MasterAGISystem()
        
        # Coding-specific state
        self.active_tasks: Dict[str, CodingTask] = {}
        self.completed_tasks: List[CodingResult] = []
        self.code_patterns = {}
        self.performance_metrics = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "average_quality": 0.0,
            "average_time": 0.0,
            "user_satisfaction": 0.0
        }
        
        # Configuration
        self.config = {
            "enable_research_mode": True,
            "enable_multi_model": True,
            "enable_self_improvement": True,
            "default_quality_level": "production",
            "security_focus": True,
            "performance_focus": True,
            "testing_focus": True,
            "documentation_focus": True,
            "code_review_enabled": True,
            "auto_optimization": True
        }
        
        # Code generation templates
        self.load_code_templates()
        
        # Start background services
        self.start_background_services()
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('advanced_coding_assistant.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def load_code_templates(self):
        """Load coding templates and patterns"""
        self.code_templates = {
            "python": {
                "web_api": {
                    "template": """
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="{api_name}", version="1.0.0")
security = HTTPBearer()

# Models
class {model_name}(BaseModel):
    # Define your model fields here
    pass

# Authentication
def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    # Implement token verification
    pass

# Routes
@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

@app.post("/{endpoint}")
def {endpoint_function}(
    data: {model_name},
    token: str = Depends(verify_token)
):
    try:
        # Implement your logic here
        return {"success": True, "data": data}
    except Exception as e:
        logger.error(f"Error in {endpoint_function}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
""",
                    "security_features": [
                        "JWT token authentication",
                        "Input validation with Pydantic",
                        "Error handling and logging",
                        "CORS protection",
                        "Rate limiting"
                    ]
                },
                
                "smart_contract": {
                    "template": """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/Pausable.sol";

contract {contract_name} is ReentrancyGuard, Ownable, Pausable {
    
    // State variables
    mapping(address => uint256) public balances;
    
    // Events
    event {event_name}(address indexed user, uint256 amount, uint256 timestamp);
    
    // Modifiers
    modifier validAmount(uint256 amount) {
        require(amount > 0, "Amount must be greater than 0");
        _;
    }
    
    // Constructor
    constructor() {
        // Initialize contract
    }
    
    // Main functions
    function {main_function}(uint256 amount) 
        external 
        nonReentrant 
        whenNotPaused 
        validAmount(amount) 
    {
        // Implement main logic
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        balances[msg.sender] -= amount;
        
        emit {event_name}(msg.sender, amount, block.timestamp);
    }
    
    // Emergency functions
    function pause() external onlyOwner {
        _pause();
    }
    
    function unpause() external onlyOwner {
        _unpause();
    }
}
""",
                    "security_features": [
                        "Reentrancy protection",
                        "Access control",
                        "Pausable functionality",
                        "Input validation",
                        "Event logging"
                    ]
                }
            }
        }
    
    def start_background_services(self):
        """Start background services for continuous improvement"""
        
        if self.config["enable_self_improvement"]:
            self.research_system.start_continuous_improvement()
        
        self.logger.info("Background services started")
    
    async def solve_coding_problem(
        self, 
        description: str, 
        context: Dict[str, Any] = None,
        preferences: Dict[str, Any] = None
    ) -> CodingResult:
        """
        Main interface for solving coding problems with full research and optimization
        """
        
        task_id = f"task_{int(time.time())}"
        start_time = time.time()
        
        self.logger.info(f"Starting coding task: {task_id}")
        
        try:
            # Phase 1: Task Analysis and Planning
            task = await self.analyze_coding_task(description, context, preferences)
            self.active_tasks[task_id] = task
            
            # Phase 2: Research and Solution Design
            if self.config["enable_research_mode"]:
                research_result = await self.research_system.enhanced_research(
                    description, context
                )
                solution_context = research_result.result
            else:
                solution_context = await self.quick_solution_research(description, task)
            
            # Phase 3: Code Generation
            code_result = await self.generate_code_solution(task, solution_context)
            
            # Phase 4: Quality Assurance
            qa_result = await self.perform_quality_assurance(code_result, task)
            
            # Phase 5: Optimization and Enhancement
            final_result = await self.optimize_and_enhance(qa_result, task)
            
            execution_time = time.time() - start_time
            
            # Create final result
            result = CodingResult(
                task_id=task_id,
                code=final_result.get("code", ""),
                explanation=final_result.get("explanation", ""),
                tests=final_result.get("tests", ""),
                documentation=final_result.get("documentation", ""),
                security_notes=final_result.get("security_notes", ""),
                performance_notes=final_result.get("performance_notes", ""),
                deployment_guide=final_result.get("deployment_guide", ""),
                quality_score=final_result.get("quality_score", 0.0),
                execution_time=execution_time,
                model_used=final_result.get("model_used", "unknown"),
                success=True,
                timestamp=datetime.now()
            )
            
            # Update metrics and learning
            await self.update_performance_metrics(result)
            self.completed_tasks.append(result)
            
            # Clean up active task
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            
            self.logger.info(f"Coding task completed: {task_id} (Quality: {result.quality_score:.3f})")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            error_result = CodingResult(
                task_id=task_id,
                code="",
                explanation=f"Error: {str(e)}",
                tests="",
                documentation="",
                security_notes="",
                performance_notes="",
                deployment_guide="",
                quality_score=0.0,
                execution_time=execution_time,
                model_used="error",
                success=False,
                timestamp=datetime.now()
            )
            
            self.completed_tasks.append(error_result)
            self.logger.error(f"Coding task failed: {task_id} - {e}")
            
            return error_result
    
    async def analyze_coding_task(
        self, 
        description: str, 
        context: Dict[str, Any] = None,
        preferences: Dict[str, Any] = None
    ) -> CodingTask:
        """Analyze and classify the coding task"""
        
        context = context or {}
        preferences = preferences or {}
        
        # Use model orchestrator to classify the task
        classification = self.model_orchestrator.classify_task(description, context)
        
        # Extract specific requirements
        requirements = self.extract_requirements(description)
        constraints = self.extract_constraints(description, preferences)
        
        # Determine language and framework
        language = self.detect_language(description, context)
        framework = self.detect_framework(description, context, language)
        
        # Create task object
        task = CodingTask(
            task_id=f"task_{int(time.time())}",
            description=description,
            language=language,
            framework=framework,
            complexity=classification.complexity_level,
            requirements=requirements,
            constraints=constraints,
            context=context,
            priority=preferences.get("priority", "medium"),
            deadline=preferences.get("deadline")
        )
        
        self.logger.info(f"Task analyzed: {language}/{framework}, complexity {task.complexity}")
        
        return task
    
    def extract_requirements(self, description: str) -> List[str]:
        """Extract functional and non-functional requirements"""
        
        requirements = []
        description_lower = description.lower()
        
        # Functional requirements
        if "authentication" in description_lower or "login" in description_lower:
            requirements.append("User authentication system")
        
        if "database" in description_lower or "store" in description_lower:
            requirements.append("Data persistence layer")
        
        if "api" in description_lower or "endpoint" in description_lower:
            requirements.append("REST API interface")
        
        if "real-time" in description_lower or "websocket" in description_lower:
            requirements.append("Real-time communication")
        
        # Non-functional requirements
        if "secure" in description_lower or "security" in description_lower:
            requirements.append("Security measures")
        
        if "performance" in description_lower or "fast" in description_lower:
            requirements.append("Performance optimization")
        
        if "scalable" in description_lower or "scale" in description_lower:
            requirements.append("Scalability design")
        
        if "test" in description_lower:
            requirements.append("Comprehensive testing")
        
        return requirements
    
    def extract_constraints(self, description: str, preferences: Dict[str, Any]) -> List[str]:
        """Extract project constraints"""
        
        constraints = []
        
        # Technical constraints
        if preferences.get("local_only", False):
            constraints.append("Local deployment only")
        
        if preferences.get("lightweight", False):
            constraints.append("Lightweight implementation")
        
        if preferences.get("budget_conscious", False):
            constraints.append("Cost-effective solutions")
        
        # Time constraints
        if preferences.get("deadline"):
            constraints.append(f"Deadline: {preferences['deadline']}")
        
        # Platform constraints
        if preferences.get("platform"):
            constraints.append(f"Platform: {preferences['platform']}")
        
        return constraints
    
    def detect_language(self, description: str, context: Dict[str, Any]) -> Optional[str]:
        """Detect the most appropriate programming language"""
        
        description_lower = description.lower()
        
        # Explicit language mentions
        language_keywords = {
            "python": ["python", "django", "flask", "fastapi", "pandas", "numpy"],
            "javascript": ["javascript", "js", "node", "react", "vue", "angular"],
            "typescript": ["typescript", "ts"],
            "solidity": ["solidity", "smart contract", "ethereum", "blockchain"],
            "rust": ["rust", "cargo"],
            "go": ["golang", "go"],
            "java": ["java", "spring"],
            "c#": ["c#", "csharp", ".net", "dotnet"],
            "cpp": ["c++", "cpp"],
            "sql": ["sql", "mysql", "postgresql", "database query"]
        }
        
        for language, keywords in language_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                return language
        
        # Context-based detection
        if context.get("language"):
            return context["language"]
        
        # Default based on task type
        if "web" in description_lower:
            return "python"  # Default to Python for web tasks
        elif "smart contract" in description_lower or "blockchain" in description_lower:
            return "solidity"
        elif "mobile" in description_lower:
            return "javascript"
        
        return "python"  # Default language
    
    def detect_framework(self, description: str, context: Dict[str, Any], language: str) -> Optional[str]:
        """Detect the most appropriate framework"""
        
        description_lower = description.lower()
        
        framework_keywords = {
            "python": {
                "fastapi": ["fastapi", "fast api", "modern api"],
                "django": ["django", "full-stack web"],
                "flask": ["flask", "lightweight web"],
                "streamlit": ["streamlit", "data app", "dashboard"]
            },
            "javascript": {
                "react": ["react", "frontend", "ui"],
                "express": ["express", "node", "backend api"],
                "nextjs": ["next.js", "nextjs", "full-stack"],
                "vue": ["vue", "vuejs"]
            },
            "solidity": {
                "hardhat": ["hardhat", "testing", "deployment"],
                "truffle": ["truffle"],
                "foundry": ["foundry", "forge"]
            }
        }
        
        if language in framework_keywords:
            for framework, keywords in framework_keywords[language].items():
                if any(keyword in description_lower for keyword in keywords):
                    return framework
        
        # Default frameworks by language
        defaults = {
            "python": "fastapi",
            "javascript": "express",
            "typescript": "express",
            "solidity": "hardhat"
        }
        
        return defaults.get(language)
    
    async def quick_solution_research(self, description: str, task: CodingTask) -> str:
        """Quick research mode for simple tasks"""
        
        research_prompt = f"""
Research and analyze the following coding task:

Task: {description}
Language: {task.language}
Framework: {task.framework}
Complexity: {task.complexity}/5

Provide:
1. **Approach**: Best approach and architecture
2. **Key Components**: Main components needed
3. **Libraries**: Recommended libraries and dependencies
4. **Patterns**: Relevant design patterns
5. **Security**: Security considerations
6. **Performance**: Performance considerations

Keep the research concise but comprehensive.
"""
        
        result = await self.model_orchestrator.intelligent_execute(research_prompt)
        return result.get('response', 'No research available')
    
    async def generate_code_solution(self, task: CodingTask, research_context: str) -> Dict[str, Any]:
        """Generate the complete code solution"""
        
        # Select template if available
        template = self.get_code_template(task)
        
        # Create comprehensive coding prompt
        coding_prompt = self.create_coding_prompt(task, research_context, template)
        
        # Generate code with optimal model
        result = await self.model_orchestrator.intelligent_execute(
            coding_prompt,
            preferences={
                "local_only": True,
                "max_response_time": 60.0
            }
        )
        
        if result['success']:
            return self.parse_code_response(result['response'], task)
        else:
            raise Exception(f"Code generation failed: {result.get('error', 'Unknown error')}")
    
    def get_code_template(self, task: CodingTask) -> Optional[Dict[str, Any]]:
        """Get appropriate code template for the task"""
        
        if task.language not in self.code_templates:
            return None
        
        language_templates = self.code_templates[task.language]
        
        # Match template based on requirements
        for template_name, template_data in language_templates.items():
            if self.matches_template(task, template_name):
                return template_data
        
        return None
    
    def matches_template(self, task: CodingTask, template_name: str) -> bool:
        """Check if task matches a template"""
        
        description_lower = task.description.lower()
        
        template_keywords = {
            "web_api": ["api", "rest", "endpoint", "web service"],
            "smart_contract": ["smart contract", "blockchain", "defi", "token"],
            "data_processing": ["data", "processing", "analysis", "etl"],
            "authentication": ["auth", "login", "jwt", "oauth"],
            "database": ["database", "crud", "orm", "sql"]
        }
        
        if template_name in template_keywords:
            keywords = template_keywords[template_name]
            return any(keyword in description_lower for keyword in keywords)
        
        return False
    
    def create_coding_prompt(self, task: CodingTask, research_context: str, template: Optional[Dict[str, Any]]) -> str:
        """Create comprehensive coding prompt"""
        
        base_prompt = f"""
You are an expert {task.language} developer. Create a complete, production-ready solution for:

**TASK**: {task.description}

**REQUIREMENTS**:
{chr(10).join(f"- {req}" for req in task.requirements)}

**CONSTRAINTS**:
{chr(10).join(f"- {constraint}" for constraint in task.constraints)}

**RESEARCH CONTEXT**:
{research_context}

**TECHNICAL SPECIFICATIONS**:
- Language: {task.language}
- Framework: {task.framework or 'Standard library'}
- Complexity Level: {task.complexity}/5
- Quality Level: {self.config['default_quality_level']}

"""
        
        if template:
            base_prompt += f"""
**TEMPLATE GUIDANCE**:
Use the following template as a starting point:
```{task.language}
{template.get('template', '')}
```

Security features to include:
{chr(10).join(f"- {feature}" for feature in template.get('security_features', []))}

"""
        
        base_prompt += f"""
**DELIVERABLES** (provide each in separate sections):

1. **COMPLETE CODE**: Full implementation with all necessary imports, error handling, and comments
2. **EXPLANATION**: Detailed explanation of the approach, architecture, and key decisions
3. **TESTS**: Comprehensive test suite covering main functionality and edge cases
4. **DOCUMENTATION**: Complete setup, configuration, and usage documentation
5. **SECURITY NOTES**: Security considerations, vulnerabilities addressed, and recommendations
6. **PERFORMANCE NOTES**: Performance optimizations, bottlenecks, and scaling considerations
7. **DEPLOYMENT GUIDE**: Step-by-step deployment and configuration instructions

**CODE QUALITY REQUIREMENTS**:
- Production-ready code with proper error handling
- Comprehensive logging and monitoring hooks
- Security best practices implemented
- Performance optimizations applied
- Complete documentation and comments
- Comprehensive test coverage
- Clean, maintainable architecture

Ensure the solution is complete, secure, performant, and ready for production deployment.
"""
        
        return base_prompt
    
    def parse_code_response(self, response: str, task: CodingTask) -> Dict[str, Any]:
        """Parse the model response into structured components"""
        
        # Initialize result structure
        result = {
            "code": "",
            "explanation": "",
            "tests": "",
            "documentation": "",
            "security_notes": "",
            "performance_notes": "",
            "deployment_guide": ""
        }
        
        # Split response into sections
        sections = self.extract_sections(response)
        
        # Map sections to result components
        section_mapping = {
            "complete code": "code",
            "code": "code",
            "implementation": "code",
            "explanation": "explanation",
            "approach": "explanation",
            "tests": "tests",
            "test": "tests",
            "testing": "tests",
            "documentation": "documentation",
            "docs": "documentation",
            "security": "security_notes",
            "security notes": "security_notes",
            "performance": "performance_notes",
            "performance notes": "performance_notes",
            "deployment": "deployment_guide",
            "deployment guide": "deployment_guide"
        }
        
        for section_title, content in sections.items():
            section_key = section_mapping.get(section_title.lower())
            if section_key:
                result[section_key] = content
        
        # Extract code blocks if no structured sections found
        if not result["code"] and "```" in response:
            code_blocks = self.extract_code_blocks(response)
            if code_blocks:
                result["code"] = code_blocks[0]  # Use first code block as main code
        
        # If still no code, use entire response as explanation
        if not result["code"] and not result["explanation"]:
            result["explanation"] = response
        
        return result
    
    def extract_sections(self, response: str) -> Dict[str, str]:
        """Extract structured sections from response"""
        
        sections = {}
        current_section = None
        current_content = []
        
        lines = response.split('\n')
        
        for line in lines:
            # Check for section headers
            if line.strip().startswith('**') and line.strip().endswith('**'):
                # Save previous section
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = line.strip().replace('*', '').strip()
                current_content = []
            
            elif line.strip().startswith('#') and current_section is None:
                # Markdown-style headers
                current_section = line.strip().replace('#', '').strip()
                current_content = []
            
            else:
                # Add to current section
                if current_section:
                    current_content.append(line)
        
        # Save last section
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def extract_code_blocks(self, response: str) -> List[str]:
        """Extract code blocks from response"""
        
        code_blocks = []
        in_code_block = False
        current_block = []
        
        lines = response.split('\n')
        
        for line in lines:
            if line.strip().startswith('```'):
                if in_code_block:
                    # End of code block
                    code_blocks.append('\n'.join(current_block))
                    current_block = []
                    in_code_block = False
                else:
                    # Start of code block
                    in_code_block = True
            elif in_code_block:
                current_block.append(line)
        
        return code_blocks
    
    async def perform_quality_assurance(self, code_result: Dict[str, Any], task: CodingTask) -> Dict[str, Any]:
        """Perform comprehensive quality assurance on the generated code"""
        
        if not self.config["code_review_enabled"]:
            return code_result
        
        qa_prompt = f"""
Perform a comprehensive code review and quality assessment of the following {task.language} code:

**CODE TO REVIEW**:
```{task.language}
{code_result.get('code', '')}
```

**TASK CONTEXT**: {task.description}

**QUALITY ASSESSMENT CRITERIA**:
1. **Code Quality**: Structure, readability, maintainability
2. **Security**: Vulnerabilities, best practices, input validation
3. **Performance**: Efficiency, optimization opportunities
4. **Error Handling**: Exception handling, edge cases
5. **Testing**: Testability, coverage considerations
6. **Documentation**: Comments, docstrings, clarity

**PROVIDE**:
1. **Quality Score**: Overall score from 0.0 to 1.0
2. **Issues Found**: List of specific issues with severity levels
3. **Improvements**: Concrete suggestions for enhancement
4. **Security Review**: Security assessment and recommendations
5. **Performance Review**: Performance analysis and optimizations

Format your response with clear sections for each assessment area.
"""
        
        qa_result = await self.model_orchestrator.intelligent_execute(qa_prompt)
        
        if qa_result['success']:
            # Parse QA results
            qa_analysis = self.parse_qa_response(qa_result['response'])
            
            # Integrate QA findings into code result
            code_result.update({
                "quality_score": qa_analysis.get("quality_score", 0.8),
                "qa_issues": qa_analysis.get("issues", []),
                "qa_improvements": qa_analysis.get("improvements", []),
                "security_assessment": qa_analysis.get("security_review", ""),
                "performance_assessment": qa_analysis.get("performance_review", "")
            })
        
        return code_result
    
    def parse_qa_response(self, response: str) -> Dict[str, Any]:
        """Parse quality assurance response"""
        
        qa_result = {
            "quality_score": 0.8,  # Default score
            "issues": [],
            "improvements": [],
            "security_review": "",
            "performance_review": ""
        }
        
        # Extract quality score
        if "quality score" in response.lower():
            # Look for numerical score
            import re
            score_match = re.search(r'(\d+\.?\d*)', response.lower().split("quality score")[1].split('\n')[0])
            if score_match:
                score = float(score_match.group(1))
                if score > 1.0:
                    score = score / 100  # Convert percentage to decimal
                qa_result["quality_score"] = min(1.0, max(0.0, score))
        
        # Extract sections using the same method as before
        sections = self.extract_sections(response)
        
        for section_title, content in sections.items():
            title_lower = section_title.lower()
            if "issue" in title_lower:
                qa_result["issues"] = content.split('\n')
            elif "improvement" in title_lower:
                qa_result["improvements"] = content.split('\n')
            elif "security" in title_lower:
                qa_result["security_review"] = content
            elif "performance" in title_lower:
                qa_result["performance_review"] = content
        
        return qa_result
    
    async def optimize_and_enhance(self, code_result: Dict[str, Any], task: CodingTask) -> Dict[str, Any]:
        """Optimize and enhance the code based on QA results"""
        
        if not self.config["auto_optimization"]:
            return code_result
        
        quality_score = code_result.get("quality_score", 1.0)
        
        # Only optimize if quality is below threshold
        if quality_score >= 0.85:
            self.logger.info(f"Code quality sufficient ({quality_score:.3f}), skipping optimization")
            return code_result
        
        self.logger.info(f"Optimizing code (current quality: {quality_score:.3f})")
        
        # Create optimization prompt
        optimization_prompt = f"""
Optimize and enhance the following {task.language} code based on the quality assessment:

**CURRENT CODE**:
```{task.language}
{code_result.get('code', '')}
```

**QUALITY ASSESSMENT**:
- Current Score: {quality_score:.3f}
- Issues: {', '.join(code_result.get('qa_issues', [])[:3])}
- Key Improvements Needed: {', '.join(code_result.get('qa_improvements', [])[:3])}

**OPTIMIZATION REQUIREMENTS**:
1. Address all critical security and performance issues
2. Improve code structure and readability
3. Enhance error handling and edge case coverage
4. Optimize for production deployment
5. Ensure comprehensive testing capabilities

**PROVIDE**:
1. **OPTIMIZED CODE**: Complete enhanced implementation
2. **IMPROVEMENTS MADE**: List of specific enhancements
3. **QUALITY JUSTIFICATION**: Why this meets production standards

Ensure the optimized code achieves at least 0.9 quality score.
"""
        
        optimization_result = await self.model_orchestrator.intelligent_execute(optimization_prompt)
        
        if optimization_result['success']:
            # Parse optimized response
            optimized_sections = self.extract_sections(optimization_result['response'])
            
            # Update code with optimized version
            if 'optimized code' in optimized_sections:
                code_result['code'] = optimized_sections['optimized code']
                code_result['quality_score'] = min(1.0, quality_score + 0.2)  # Boost quality score
            
            # Add optimization notes
            if 'improvements made' in optimized_sections:
                code_result['optimization_notes'] = optimized_sections['improvements made']
        
        return code_result
    
    async def update_performance_metrics(self, result: CodingResult):
        """Update system performance metrics"""
        
        self.performance_metrics["total_tasks"] += 1
        
        if result.success:
            self.performance_metrics["successful_tasks"] += 1
        
        # Update averages
        total = self.performance_metrics["total_tasks"]
        
        # Quality average
        current_avg_quality = self.performance_metrics["average_quality"]
        self.performance_metrics["average_quality"] = (
            (current_avg_quality * (total - 1) + result.quality_score) / total
        )
        
        # Time average
        current_avg_time = self.performance_metrics["average_time"]
        self.performance_metrics["average_time"] = (
            (current_avg_time * (total - 1) + result.execution_time) / total
        )
        
        # Save metrics
        with open("coding_assistant_metrics.json", 'w') as f:
            json.dump(self.performance_metrics, f, indent=2)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        return {
            "performance_metrics": self.performance_metrics,
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "config": self.config,
            "components_status": {
                "research_system": self.research_system.get_system_status(),
                "model_orchestrator": "operational",
                "agi_system": "operational"
            },
            "supported_languages": list(self.code_templates.keys()),
            "system_health": "optimal" if self.performance_metrics["average_quality"] > 0.8 else "needs_improvement"
        }
    
    async def interactive_coding_session(self):
        """Start an interactive coding session"""
        
        print("🚀 Advanced Coding Assistant - Interactive Session")
        print("=" * 60)
        print("Enter your coding tasks. Type 'exit' to quit, 'status' for system status.")
        print()
        
        while True:
            try:
                # Get user input
                user_input = input("\n💻 Describe your coding task: ").strip()
                
                if user_input.lower() == 'exit':
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'status':
                    status = self.get_system_status()
                    print(f"\n📊 System Status:")
                    print(f"  Quality Average: {status['performance_metrics']['average_quality']:.3f}")
                    print(f"  Success Rate: {status['performance_metrics']['successful_tasks']}/{status['performance_metrics']['total_tasks']}")
                    print(f"  System Health: {status['system_health']}")
                    continue
                
                if not user_input:
                    continue
                
                print(f"\n🔍 Analyzing task...")
                
                # Process the coding task
                result = await self.solve_coding_problem(user_input)
                
                if result.success:
                    print(f"\n✅ Task completed successfully!")
                    print(f"⏱️ Execution time: {result.execution_time:.1f}s")
                    print(f"🎯 Quality score: {result.quality_score:.3f}")
                    print(f"🤖 Model used: {result.model_used}")
                    
                    # Show code preview
                    if result.code:
                        print(f"\n📝 Generated code preview:")
                        code_preview = result.code[:500] + "..." if len(result.code) > 500 else result.code
                        print(f"```{code_preview}```")
                    
                    # Save full result
                    output_file = f"coding_result_{result.task_id}.json"
                    with open(output_file, 'w') as f:
                        json.dump(asdict(result), f, indent=2, default=str)
                    print(f"💾 Full result saved to: {output_file}")
                
                else:
                    print(f"\n❌ Task failed: {result.explanation}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n⚠️ Error: {e}")

# Global instance
coding_assistant = None

def get_coding_assistant() -> AdvancedCodingAssistant:
    """Get global coding assistant instance"""
    global coding_assistant
    if coding_assistant is None:
        coding_assistant = AdvancedCodingAssistant()
    return coding_assistant

async def main():
    """Main function for testing the advanced coding assistant"""
    
    assistant = AdvancedCodingAssistant()
    
    # Run interactive session
    await assistant.interactive_coding_session()

if __name__ == "__main__":
    asyncio.run(main())
