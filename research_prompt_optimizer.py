#!/usr/bin/env python3
"""
Research Prompt Optimizer
========================
Advanced prompt engineering system that creates optimized research prompts
for building a local Claude/Codex-type coding assistant with self-improvement.
"""

import json
import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import requests

@dataclass
class ResearchPromptConfig:
    """Configuration for research-optimized prompts"""
    task_type: str
    domain: str
    complexity_level: int  # 1-5 scale
    research_depth: str   # "shallow", "medium", "deep"
    code_quality_level: str  # "prototype", "production", "enterprise"
    security_focus: bool
    performance_focus: bool
    maintainability_focus: bool
    testing_focus: bool

class ResearchPromptOptimizer:
    """Optimizes prompts for research-first coding assistance"""
    
    def __init__(self):
        self.setup_logging()
        self.load_base_templates()
        self.performance_history = []
        
    def setup_logging(self):
        """Setup logging for prompt optimization"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('research_prompt_optimizer.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def load_base_templates(self):
        """Load base prompt templates for different scenarios"""
        self.base_templates = {
            "research_first_system": """You are an elite research-driven software engineer and architect. Before providing any solution, you:

1. **RESEARCH PHASE**: Analyze the problem deeply
   - Identify the core requirements and constraints
   - Research current best practices and industry standards
   - Consider security, performance, and scalability implications
   - Evaluate multiple solution approaches

2. **DESIGN PHASE**: Architect the optimal solution
   - Choose the most appropriate patterns and technologies
   - Design for maintainability and extensibility
   - Plan for testing and deployment considerations
   - Consider edge cases and error handling

3. **IMPLEMENTATION PHASE**: Provide production-ready code
   - Write clean, well-documented, secure code
   - Include comprehensive error handling
   - Add performance optimizations where relevant
   - Provide testing strategies and examples

4. **VALIDATION PHASE**: Ensure quality and completeness
   - Review security vulnerabilities
   - Validate performance characteristics
   - Confirm maintainability standards
   - Suggest monitoring and observability

Always provide complete, production-ready solutions with detailed explanations.""",

            "code_generation": """You are a master code generator specializing in {domain}. Your approach:

**ANALYSIS**: 
- Understand the exact requirements and context
- Research best practices for {domain}
- Identify potential challenges and edge cases

**ARCHITECTURE**:
- Design scalable, maintainable solutions
- Choose appropriate patterns and libraries
- Plan for testing and error handling

**IMPLEMENTATION**:
- Generate clean, readable, well-documented code
- Include comprehensive error handling and validation
- Implement proper logging and monitoring hooks
- Follow {domain} security best practices

**QUALITY ASSURANCE**:
- Provide unit test examples
- Include performance considerations
- Add security review notes
- Suggest deployment strategies

Focus on {code_quality_level} quality standards.""",

            "architecture_design": """You are a senior software architect specializing in {domain} systems. Your methodology:

**REQUIREMENTS ANALYSIS**:
- Deep dive into functional and non-functional requirements
- Identify scalability, performance, and security constraints
- Research industry standards and compliance needs

**SOLUTION ARCHITECTURE**:
- Design modular, scalable architecture
- Choose appropriate technologies and patterns
- Plan for fault tolerance and disaster recovery
- Consider monitoring, logging, and observability

**IMPLEMENTATION STRATEGY**:
- Break down into implementable components
- Define clear interfaces and contracts
- Plan deployment and rollout strategy
- Design comprehensive testing approach

**RISK MITIGATION**:
- Identify potential failure points
- Plan for security vulnerabilities
- Design for operational excellence
- Include performance optimization strategies

Deliver enterprise-grade architectural solutions.""",

            "problem_solving": """You are an expert problem solver with deep research capabilities. Your process:

**PROBLEM RESEARCH**:
- Thoroughly understand the problem domain
- Research existing solutions and their limitations
- Identify similar problems and their solutions
- Analyze root causes and contributing factors

**SOLUTION EXPLORATION**:
- Generate multiple solution approaches
- Evaluate pros and cons of each approach
- Research implementation complexity and risks
- Consider long-term maintenance implications

**OPTIMAL SOLUTION**:
- Select the best approach based on research
- Design implementation plan with clear steps
- Include fallback strategies and error handling
- Provide monitoring and validation methods

**IMPLEMENTATION GUIDANCE**:
- Detailed step-by-step implementation
- Code examples with best practices
- Testing and validation strategies
- Performance and security considerations

Always base solutions on thorough research and proven practices.""",

            "security_first": """You are a security-focused software engineer. Your security-first approach:

**THREAT MODELING**:
- Identify potential attack vectors and vulnerabilities
- Research current security threats in {domain}
- Analyze compliance requirements and standards
- Evaluate risk levels and impact assessment

**SECURE DESIGN**:
- Design with security by default principles
- Implement defense in depth strategies
- Use proven security patterns and libraries
- Plan for secure deployment and operations

**SECURE IMPLEMENTATION**:
- Write code following security best practices
- Implement proper input validation and sanitization
- Use secure authentication and authorization
- Include comprehensive logging for security events

**SECURITY VALIDATION**:
- Provide security testing strategies
- Include penetration testing considerations
- Plan for security monitoring and incident response
- Design for security updates and patches

Never compromise security for convenience or performance.""",

            "performance_optimization": """You are a performance optimization specialist. Your optimization methodology:

**PERFORMANCE RESEARCH**:
- Analyze performance requirements and constraints
- Research performance patterns in {domain}
- Identify common performance bottlenecks
- Study optimization techniques and their trade-offs

**PERFORMANCE DESIGN**:
- Design for performance from the ground up
- Choose optimal algorithms and data structures
- Plan for efficient resource utilization
- Design performance monitoring strategies

**OPTIMIZATION IMPLEMENTATION**:
- Write performance-optimized code
- Implement efficient caching strategies
- Use asynchronous patterns where appropriate
- Include performance profiling hooks

**PERFORMANCE VALIDATION**:
- Provide performance testing strategies
- Include benchmarking methodologies
- Plan for performance monitoring in production
- Design for performance scaling and optimization

Always balance performance with maintainability and security."""
        }
    
    def generate_optimized_prompt(self, config: ResearchPromptConfig) -> str:
        """Generate an optimized prompt based on configuration"""
        
        # Select base template
        if config.security_focus:
            base_template = self.base_templates["security_first"]
        elif config.performance_focus:
            base_template = self.base_templates["performance_optimization"]
        elif config.task_type == "architecture":
            base_template = self.base_templates["architecture_design"]
        elif config.task_type == "problem_solving":
            base_template = self.base_templates["problem_solving"]
        else:
            base_template = self.base_templates["code_generation"]
        
        # Customize based on configuration
        prompt = base_template.format(
            domain=config.domain,
            code_quality_level=config.code_quality_level
        )
        
        # Add complexity-specific instructions
        if config.complexity_level >= 4:
            prompt += f"\n\n**HIGH COMPLEXITY REQUIREMENTS**:\n"
            prompt += "- Consider distributed systems patterns\n"
            prompt += "- Plan for horizontal scaling\n"
            prompt += "- Include comprehensive error recovery\n"
            prompt += "- Design for operational excellence\n"
        
        # Add research depth instructions
        if config.research_depth == "deep":
            prompt += f"\n\n**DEEP RESEARCH REQUIREMENTS**:\n"
            prompt += "- Research latest industry trends and innovations\n"
            prompt += "- Analyze multiple implementation approaches\n"
            prompt += "- Include comparative analysis of solutions\n"
            prompt += "- Provide detailed rationale for choices\n"
        
        # Add domain-specific enhancements
        domain_enhancements = self.get_domain_enhancements(config.domain)
        if domain_enhancements:
            prompt += f"\n\n**{config.domain.upper()} SPECIFIC REQUIREMENTS**:\n"
            prompt += domain_enhancements
        
        # Add quality level requirements
        quality_requirements = self.get_quality_requirements(config.code_quality_level)
        prompt += f"\n\n**{config.code_quality_level.upper()} QUALITY STANDARDS**:\n"
        prompt += quality_requirements
        
        return prompt
    
    def get_domain_enhancements(self, domain: str) -> str:
        """Get domain-specific enhancements"""
        domain_specs = {
            "web_development": """- Follow modern web security practices (OWASP guidelines)
- Implement responsive design principles
- Use progressive enhancement strategies
- Include accessibility (WCAG) compliance
- Plan for SEO optimization
- Consider performance metrics (Core Web Vitals)""",
            
            "blockchain": """- Follow smart contract security best practices
- Implement gas optimization strategies
- Use established patterns (OpenZeppelin, etc.)
- Include comprehensive testing (unit, integration, fuzzing)
- Plan for upgradability and governance
- Consider MEV protection and front-running prevention""",
            
            "ai_ml": """- Follow ML engineering best practices
- Implement model versioning and lineage tracking
- Use proper data validation and monitoring
- Include experiment tracking and reproducibility
- Plan for model deployment and serving
- Consider bias detection and fairness metrics""",
            
            "devops": """- Follow Infrastructure as Code principles
- Implement GitOps workflows
- Use container security best practices
- Include comprehensive monitoring and observability
- Plan for disaster recovery and backup strategies
- Consider cost optimization and resource management""",
            
            "api_development": """- Follow RESTful API design principles
- Implement proper authentication and authorization
- Use API versioning strategies
- Include comprehensive documentation (OpenAPI)
- Plan for rate limiting and throttling
- Consider caching and performance optimization"""
        }
        
        return domain_specs.get(domain, "- Follow industry best practices and standards")
    
    def get_quality_requirements(self, quality_level: str) -> str:
        """Get quality level specific requirements"""
        quality_specs = {
            "prototype": """- Focus on rapid development and validation
- Include basic error handling
- Provide clear documentation of assumptions
- Plan for future enhancement and refactoring""",
            
            "production": """- Implement comprehensive error handling and logging
- Include unit and integration tests
- Follow coding standards and best practices
- Plan for monitoring and maintenance
- Include deployment and rollback strategies""",
            
            "enterprise": """- Implement enterprise-grade security measures
- Include comprehensive testing (unit, integration, e2e, security)
- Follow strict coding standards and review processes
- Plan for high availability and disaster recovery
- Include comprehensive monitoring, alerting, and observability
- Design for compliance and audit requirements
- Plan for scaling and performance optimization"""
        }
        
        return quality_specs.get(quality_level, quality_specs["production"])
    
    def optimize_for_cline_model(self, prompt: str, model_name: str = "cline-optimal") -> str:
        """Optimize prompt specifically for the cline-optimal model"""
        
        # Add cline-specific optimizations
        cline_optimizations = f"""

**CLINE-OPTIMAL SPECIFIC INSTRUCTIONS**:
- Provide complete, executable solutions
- Include all necessary imports and dependencies
- Use clear, descriptive variable and function names
- Add comprehensive comments and docstrings
- Structure code for maximum readability
- Include error handling for edge cases
- Provide usage examples and test cases

**OUTPUT FORMAT**:
1. **Research Summary**: Brief analysis of the problem and approach
2. **Architecture Overview**: High-level design and patterns used
3. **Implementation**: Complete, production-ready code
4. **Testing Strategy**: Unit tests and validation approaches
5. **Deployment Notes**: Installation and configuration guidance
6. **Monitoring**: Observability and maintenance considerations

**CODE QUALITY CHECKLIST**:
✓ All imports clearly specified
✓ Functions have type hints where appropriate
✓ Comprehensive error handling implemented
✓ Security considerations addressed
✓ Performance implications noted
✓ Testing approach included
✓ Documentation complete"""

        return prompt + cline_optimizations
    
    def create_research_session(self, task_description: str, config: ResearchPromptConfig) -> Dict[str, Any]:
        """Create a complete research session with optimized prompts"""
        
        session_id = f"research_{int(time.time())}"
        
        # Generate the main research prompt
        main_prompt = self.generate_optimized_prompt(config)
        optimized_prompt = self.optimize_for_cline_model(main_prompt)
        
        # Create complete prompt for the task
        complete_prompt = f"""{optimized_prompt}

**RESEARCH TASK**: {task_description}

Please provide a comprehensive, research-driven solution following the methodology outlined above."""
        
        session = {
            "session_id": session_id,
            "task_description": task_description,
            "config": asdict(config),
            "generated_prompt": complete_prompt,
            "timestamp": datetime.now().isoformat(),
            "status": "ready"
        }
        
        # Save session for tracking
        self.save_research_session(session)
        
        return session
    
    def save_research_session(self, session: Dict[str, Any]):
        """Save research session for tracking and optimization"""
        os.makedirs("research_sessions", exist_ok=True)
        
        filename = f"research_sessions/session_{session['session_id']}.json"
        with open(filename, 'w') as f:
            json.dump(session, f, indent=2)
        
        self.logger.info(f"Saved research session: {filename}")
    
    def execute_research_with_cline(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """Execute research session with cline-optimal model"""
        
        try:
            import ollama
            
            self.logger.info(f"Executing research session: {session['session_id']}")
            
            start_time = time.time()
            
            # Execute with cline-optimal
            response = ollama.chat(
                model='cline-optimal:latest',
                messages=[
                    {
                        'role': 'user',
                        'content': session['generated_prompt']
                    }
                ],
                options={
                    'temperature': 0.3,  # Lower temperature for more focused research
                    'top_p': 0.9,
                    'max_tokens': 4000
                }
            )
            
            execution_time = time.time() - start_time
            
            # Parse response
            result = {
                "session_id": session['session_id'],
                "response": response['message']['content'],
                "execution_time": execution_time,
                "model_used": "cline-optimal:latest",
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
            # Save result
            self.save_research_result(result)
            
            # Update performance tracking
            self.update_performance_metrics(session, result)
            
            return result
            
        except Exception as e:
            error_result = {
                "session_id": session['session_id'],
                "error": str(e),
                "execution_time": 0,
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
            
            self.logger.error(f"Research execution failed: {e}")
            return error_result
    
    def save_research_result(self, result: Dict[str, Any]):
        """Save research result for analysis"""
        os.makedirs("research_results", exist_ok=True)
        
        filename = f"research_results/result_{result['session_id']}.json"
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        
        self.logger.info(f"Saved research result: {filename}")
    
    def update_performance_metrics(self, session: Dict[str, Any], result: Dict[str, Any]):
        """Update performance metrics for continuous improvement"""
        
        metrics = {
            "session_id": session['session_id'],
            "task_complexity": session['config']['complexity_level'],
            "research_depth": session['config']['research_depth'],
            "execution_time": result['execution_time'],
            "success": result['success'],
            "response_length": len(result.get('response', '')),
            "timestamp": datetime.now().isoformat()
        }
        
        self.performance_history.append(metrics)
        
        # Save performance history
        with open("research_performance_history.json", 'w') as f:
            json.dump(self.performance_history, f, indent=2)
    
    def analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends for optimization"""
        
        if not self.performance_history:
            return {"message": "No performance data available"}
        
        # Calculate metrics
        successful_sessions = [s for s in self.performance_history if s['success']]
        success_rate = len(successful_sessions) / len(self.performance_history)
        
        avg_execution_time = sum(s['execution_time'] for s in successful_sessions) / len(successful_sessions) if successful_sessions else 0
        
        complexity_performance = {}
        for level in range(1, 6):
            level_sessions = [s for s in successful_sessions if s['task_complexity'] == level]
            if level_sessions:
                complexity_performance[f"level_{level}"] = {
                    "count": len(level_sessions),
                    "avg_time": sum(s['execution_time'] for s in level_sessions) / len(level_sessions),
                    "avg_response_length": sum(s['response_length'] for s in level_sessions) / len(level_sessions)
                }
        
        return {
            "total_sessions": len(self.performance_history),
            "success_rate": success_rate,
            "avg_execution_time": avg_execution_time,
            "complexity_performance": complexity_performance,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def optimize_prompts_based_on_performance(self) -> Dict[str, Any]:
        """Optimize prompts based on performance analysis"""
        
        analysis = self.analyze_performance_trends()
        
        optimizations = {
            "identified_improvements": [],
            "recommended_changes": [],
            "next_experiments": []
        }
        
        # Analyze success rates by complexity
        if 'complexity_performance' in analysis:
            for level, perf in analysis['complexity_performance'].items():
                if perf['avg_time'] > 30:  # Slow responses
                    optimizations["identified_improvements"].append(
                        f"Complexity {level}: Execution time too high ({perf['avg_time']:.1f}s)"
                    )
                    optimizations["recommended_changes"].append(
                        f"Reduce prompt complexity for {level} tasks"
                    )
        
        # Suggest experiments
        if analysis['success_rate'] < 0.9:
            optimizations["next_experiments"].append("A/B test simplified prompt structures")
        
        if analysis['avg_execution_time'] > 20:
            optimizations["next_experiments"].append("Test temperature and token limit optimization")
        
        return optimizations

def create_research_assistant() -> ResearchPromptOptimizer:
    """Create and configure the research assistant"""
    return ResearchPromptOptimizer()

def main():
    """Main function for testing the research prompt optimizer"""
    
    print("🔬 Research Prompt Optimizer for Local Claude/Codex")
    print("=" * 60)
    
    # Create optimizer
    optimizer = ResearchPromptOptimizer()
    
    # Example configuration
    config = ResearchPromptConfig(
        task_type="code_generation",
        domain="web_development",
        complexity_level=3,
        research_depth="medium",
        code_quality_level="production",
        security_focus=True,
        performance_focus=True,
        maintainability_focus=True,
        testing_focus=True
    )
    
    # Create research session
    task = "Create a secure REST API for user authentication with JWT tokens, rate limiting, and comprehensive error handling"
    
    session = optimizer.create_research_session(task, config)
    print(f"Created research session: {session['session_id']}")
    
    # Execute research
    result = optimizer.execute_research_with_cline(session)
    
    if result['success']:
        print(f"✅ Research completed in {result['execution_time']:.1f}s")
        print(f"📝 Response length: {len(result['response'])} characters")
    else:
        print(f"❌ Research failed: {result['error']}")
    
    # Analyze performance
    performance = optimizer.analyze_performance_trends()
    print(f"📊 Success rate: {performance.get('success_rate', 0)*100:.1f}%")

if __name__ == "__main__":
    main()
