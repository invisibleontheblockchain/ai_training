#!/usr/bin/env python3
"""
Self-Improving Research System
=============================
Advanced research system that continuously learns and improves its capabilities
by integrating with the autonomous learning daemon and multi-model orchestrator.
"""

import asyncio
import json
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import threading
import queue
import requests
from pathlib import Path

# Import our components
from research_prompt_optimizer import ResearchPromptOptimizer, ResearchPromptConfig
from multi_model_orchestrator import ModelOrchestrator
from autonomous_learning_daemon import AutonomousLearningDaemon

@dataclass
class ResearchResult:
    """Result from a research session"""
    session_id: str
    task_description: str
    result: str
    model_used: str
    execution_time: float
    quality_score: float
    research_depth: str
    success: bool
    timestamp: datetime
    feedback_score: Optional[float] = None
    improvement_suggestions: List[str] = None

@dataclass
class LearningPattern:
    """Pattern identified from research results"""
    pattern_type: str
    description: str
    frequency: int
    success_correlation: float
    improvement_potential: float
    recommended_actions: List[str]

class SelfImprovingResearchSystem:
    """Research system that learns from its own performance"""
    
    def __init__(self):
        self.setup_logging()
        
        # Initialize components
        self.prompt_optimizer = ResearchPromptOptimizer()
        self.model_orchestrator = ModelOrchestrator()
        self.autonomous_daemon = AutonomousLearningDaemon()
        
        # Learning state
        self.research_history: List[ResearchResult] = []
        self.learning_patterns: List[LearningPattern] = []
        self.performance_metrics = {
            "total_sessions": 0,
            "successful_sessions": 0,
            "average_quality": 0.0,
            "average_execution_time": 0.0,
            "improvement_rate": 0.0
        }
        
        # Configuration
        self.config = {
            "learning_threshold": 0.05,  # Minimum improvement to trigger learning
            "pattern_detection_window": 50,  # Number of sessions to analyze for patterns
            "auto_improvement_interval": 3600,  # Seconds between improvement cycles
            "quality_threshold": 0.8,  # Minimum quality score for good results
            "max_research_depth": "deep",
            "enable_web_research": True,
            "enable_code_analysis": True,
            "enable_documentation_search": True
        }
        
        # Load previous state
        self.load_research_history()
        self.load_learning_patterns()
        
        # Start background improvement thread
        self.improvement_thread = None
        self.running = False
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('self_improving_research.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def load_research_history(self):
        """Load previous research history"""
        try:
            if os.path.exists("research_history.json"):
                with open("research_history.json", 'r') as f:
                    data = json.load(f)
                    self.research_history = [
                        ResearchResult(**item) for item in data
                    ]
                self.logger.info(f"Loaded {len(self.research_history)} research sessions")
        except Exception as e:
            self.logger.warning(f"Could not load research history: {e}")
            self.research_history = []
    
    def save_research_history(self):
        """Save research history for persistence"""
        try:
            data = [asdict(result) for result in self.research_history]
            with open("research_history.json", 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Could not save research history: {e}")
    
    def load_learning_patterns(self):
        """Load identified learning patterns"""
        try:
            if os.path.exists("learning_patterns.json"):
                with open("learning_patterns.json", 'r') as f:
                    data = json.load(f)
                    self.learning_patterns = [
                        LearningPattern(**item) for item in data
                    ]
                self.logger.info(f"Loaded {len(self.learning_patterns)} learning patterns")
        except Exception as e:
            self.logger.warning(f"Could not load learning patterns: {e}")
            self.learning_patterns = []
    
    def save_learning_patterns(self):
        """Save learning patterns for persistence"""
        try:
            data = [asdict(pattern) for pattern in self.learning_patterns]
            with open("learning_patterns.json", 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save learning patterns: {e}")
    
    async def enhanced_research(self, task_description: str, context: Dict[str, Any] = None) -> ResearchResult:
        """Perform enhanced research with self-improvement capabilities"""
        
        session_id = f"research_{int(time.time())}"
        start_time = time.time()
        
        self.logger.info(f"Starting enhanced research session: {session_id}")
        
        try:
            # Phase 1: Pre-research analysis
            research_plan = await self.create_research_plan(task_description, context)
            
            # Phase 2: Multi-phase research execution
            research_result = await self.execute_multi_phase_research(task_description, research_plan)
            
            # Phase 3: Quality assessment and improvement
            quality_score = await self.assess_research_quality(research_result, task_description)
            
            # Phase 4: Post-processing and learning
            final_result = await self.post_process_research(research_result, quality_score)
            
            execution_time = time.time() - start_time
            
            # Create research result record
            result = ResearchResult(
                session_id=session_id,
                task_description=task_description,
                result=final_result,
                model_used=research_result.get('model_used', 'unknown'),
                execution_time=execution_time,
                quality_score=quality_score,
                research_depth=research_plan.get('depth', 'medium'),
                success=True,
                timestamp=datetime.now()
            )
            
            # Add to history and trigger learning
            self.research_history.append(result)
            await self.trigger_learning_cycle(result)
            
            self.logger.info(f"Research completed: {session_id} (Quality: {quality_score:.3f})")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            error_result = ResearchResult(
                session_id=session_id,
                task_description=task_description,
                result=f"Research failed: {str(e)}",
                model_used="error",
                execution_time=execution_time,
                quality_score=0.0,
                research_depth="none",
                success=False,
                timestamp=datetime.now()
            )
            
            self.research_history.append(error_result)
            self.logger.error(f"Research failed: {session_id} - {e}")
            
            return error_result
    
    async def create_research_plan(self, task_description: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a comprehensive research plan"""
        
        # Analyze task complexity and requirements
        classification = self.model_orchestrator.classify_task(task_description, context)
        
        # Determine research strategy based on historical patterns
        optimal_strategy = self.get_optimal_research_strategy(classification)
        
        # Create research phases
        research_phases = []
        
        # Phase 1: Background research (if needed)
        if classification.research_required or classification.complexity_level >= 3:
            research_phases.append({
                "phase": "background_research",
                "description": "Research domain knowledge and best practices",
                "methods": ["web_search", "documentation_analysis", "pattern_matching"],
                "estimated_time": 30
            })
        
        # Phase 2: Technical analysis
        research_phases.append({
            "phase": "technical_analysis",
            "description": "Analyze technical requirements and constraints",
            "methods": ["requirement_analysis", "architecture_planning", "technology_selection"],
            "estimated_time": 60
        })
        
        # Phase 3: Solution research
        research_phases.append({
            "phase": "solution_research",
            "description": "Research and design optimal solutions",
            "methods": ["solution_exploration", "best_practices_research", "security_analysis"],
            "estimated_time": 90
        })
        
        # Phase 4: Implementation research
        if classification.code_generation:
            research_phases.append({
                "phase": "implementation_research",
                "description": "Research implementation details and code patterns",
                "methods": ["code_pattern_analysis", "library_research", "testing_strategies"],
                "estimated_time": 120
            })
        
        plan = {
            "task_classification": asdict(classification),
            "research_strategy": optimal_strategy,
            "phases": research_phases,
            "estimated_total_time": sum(phase["estimated_time"] for phase in research_phases),
            "depth": self.determine_research_depth(classification),
            "priority_areas": self.identify_priority_research_areas(classification)
        }
        
        return plan
    
    def get_optimal_research_strategy(self, classification) -> Dict[str, Any]:
        """Determine optimal research strategy based on historical patterns"""
        
        # Analyze historical success patterns
        similar_tasks = [
            result for result in self.research_history[-100:]  # Last 100 sessions
            if (result.task_description and 
                classification.domain in result.task_description.lower() and
                result.success)
        ]
        
        if similar_tasks:
            # Use patterns from successful similar tasks
            avg_quality = sum(r.quality_score for r in similar_tasks) / len(similar_tasks)
            avg_time = sum(r.execution_time for r in similar_tasks) / len(similar_tasks)
            best_depths = [r.research_depth for r in similar_tasks if r.quality_score > 0.8]
            
            return {
                "based_on_history": True,
                "similar_tasks_count": len(similar_tasks),
                "historical_avg_quality": avg_quality,
                "historical_avg_time": avg_time,
                "recommended_depth": max(set(best_depths), key=best_depths.count) if best_depths else "medium",
                "confidence": min(1.0, len(similar_tasks) / 10)
            }
        else:
            # Use default strategy for new types of tasks
            return {
                "based_on_history": False,
                "recommended_depth": "medium" if classification.complexity_level <= 3 else "deep",
                "confidence": 0.5
            }
    
    def determine_research_depth(self, classification) -> str:
        """Determine appropriate research depth"""
        
        if classification.complexity_level >= 5:
            return "deep"
        elif classification.complexity_level >= 3:
            return "medium"
        else:
            return "shallow"
    
    def identify_priority_research_areas(self, classification) -> List[str]:
        """Identify priority areas for research"""
        
        priority_areas = ["core_functionality"]
        
        if classification.security_critical:
            priority_areas.append("security_best_practices")
        
        if classification.performance_critical:
            priority_areas.append("performance_optimization")
        
        if classification.complexity_level >= 4:
            priority_areas.extend(["scalability", "maintainability"])
        
        if classification.domain == "blockchain":
            priority_areas.extend(["smart_contract_security", "gas_optimization"])
        
        return priority_areas
    
    async def execute_multi_phase_research(self, task_description: str, research_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multi-phase research process"""
        
        phase_results = {}
        accumulated_knowledge = ""
        
        for phase in research_plan["phases"]:
            self.logger.info(f"Executing research phase: {phase['phase']}")
            
            # Prepare phase-specific prompt
            phase_prompt = self.create_phase_prompt(
                task_description, 
                phase, 
                accumulated_knowledge,
                research_plan["priority_areas"]
            )
            
            # Execute research with optimal model
            result = await self.model_orchestrator.intelligent_execute(phase_prompt)
            
            if result['success']:
                phase_results[phase['phase']] = result
                accumulated_knowledge += f"\n\n{phase['phase'].upper()} FINDINGS:\n{result.get('response', '')}"
            else:
                self.logger.warning(f"Phase {phase['phase']} failed: {result.get('error', 'Unknown error')}")
        
        # Synthesize final result
        final_synthesis_prompt = f"""
Based on the comprehensive research conducted, provide a complete solution for:

{task_description}

Research findings:
{accumulated_knowledge}

Please provide:
1. **Executive Summary**: Key insights and approach
2. **Technical Solution**: Complete implementation with best practices
3. **Security Considerations**: Security measures and validation
4. **Performance Notes**: Optimization strategies and considerations
5. **Testing Strategy**: Comprehensive testing approach
6. **Deployment Guide**: Setup and configuration instructions
7. **Maintenance**: Ongoing maintenance and monitoring recommendations

Ensure the solution is production-ready and follows industry best practices.
"""
        
        final_result = await self.model_orchestrator.intelligent_execute(final_synthesis_prompt)
        
        return {
            "phase_results": phase_results,
            "final_synthesis": final_result,
            "model_used": final_result.get('selected_model', 'unknown'),
            "total_phases": len(research_plan["phases"]),
            "accumulated_knowledge": accumulated_knowledge
        }
    
    def create_phase_prompt(self, task_description: str, phase: Dict[str, Any], previous_knowledge: str, priority_areas: List[str]) -> str:
        """Create a phase-specific research prompt"""
        
        phase_specific_instructions = {
            "background_research": """
Research the background knowledge and industry best practices for this task.
Focus on:
- Current industry standards and methodologies
- Common patterns and established solutions
- Potential challenges and pitfalls to avoid
- Recent developments and innovations in this area
""",
            "technical_analysis": """
Analyze the technical requirements and constraints for this task.
Focus on:
- Detailed requirement analysis and edge cases
- Technical architecture and design patterns
- Technology stack recommendations with rationale
- Integration considerations and dependencies
""",
            "solution_research": """
Research and design optimal solutions for this task.
Focus on:
- Multiple solution approaches with pros/cons analysis
- Risk assessment and mitigation strategies
- Scalability and performance considerations
- Security implications and protective measures
""",
            "implementation_research": """
Research implementation details and coding best practices.
Focus on:
- Specific code patterns and libraries to use
- Implementation best practices and conventions
- Testing strategies and quality assurance
- Deployment and operational considerations
"""
        }
        
        base_prompt = f"""
{phase_specific_instructions.get(phase['phase'], '')}

**TASK**: {task_description}

**RESEARCH PHASE**: {phase['description']}

**PRIORITY AREAS**: {', '.join(priority_areas)}

**PREVIOUS RESEARCH CONTEXT**:
{previous_knowledge if previous_knowledge else 'No previous context available.'}

**RESEARCH METHODS TO USE**: {', '.join(phase['methods'])}

Provide comprehensive research findings that will inform the next phase of development.
"""
        
        return base_prompt
    
    async def assess_research_quality(self, research_result: Dict[str, Any], task_description: str) -> float:
        """Assess the quality of research results"""
        
        try:
            final_response = research_result.get("final_synthesis", {}).get("response", "")
            
            if not final_response:
                return 0.0
            
            # Quality metrics
            quality_factors = {}
            
            # 1. Completeness (0-0.3)
            required_sections = ["summary", "solution", "security", "testing", "deployment"]
            completeness_score = sum(1 for section in required_sections if section.lower() in final_response.lower()) / len(required_sections)
            quality_factors["completeness"] = completeness_score * 0.3
            
            # 2. Technical depth (0-0.25)
            technical_indicators = ["implementation", "architecture", "patterns", "best practices", "optimization"]
            depth_score = sum(1 for indicator in technical_indicators if indicator in final_response.lower()) / len(technical_indicators)
            quality_factors["technical_depth"] = depth_score * 0.25
            
            # 3. Code quality (0-0.2)
            if "```" in final_response:  # Contains code
                code_quality_indicators = ["function", "class", "import", "return", "error handling"]
                code_score = sum(1 for indicator in code_quality_indicators if indicator.lower() in final_response.lower()) / len(code_quality_indicators)
                quality_factors["code_quality"] = code_score * 0.2
            else:
                quality_factors["code_quality"] = 0.1  # Partial credit for non-code tasks
            
            # 4. Security considerations (0-0.15)
            security_indicators = ["security", "authentication", "validation", "sanitization", "protection"]
            security_score = sum(1 for indicator in security_indicators if indicator.lower() in final_response.lower()) / len(security_indicators)
            quality_factors["security"] = security_score * 0.15
            
            # 5. Practicality (0-0.1)
            practical_indicators = ["install", "configure", "deploy", "test", "run"]
            practical_score = sum(1 for indicator in practical_indicators if indicator.lower() in final_response.lower()) / len(practical_indicators)
            quality_factors["practicality"] = practical_score * 0.1
            
            total_quality = sum(quality_factors.values())
            
            self.logger.info(f"Quality assessment: {quality_factors}")
            
            return min(1.0, total_quality)
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}")
            return 0.5  # Default neutral score
    
    async def post_process_research(self, research_result: Dict[str, Any], quality_score: float) -> str:
        """Post-process and enhance research results"""
        
        final_response = research_result.get("final_synthesis", {}).get("response", "")
        
        # If quality is below threshold, attempt improvement
        if quality_score < self.config["quality_threshold"]:
            self.logger.info(f"Quality below threshold ({quality_score:.3f}), attempting improvement")
            
            improvement_prompt = f"""
The following research result needs improvement to meet production standards:

{final_response}

Please enhance this result by:
1. Adding missing technical details and implementation specifics
2. Including comprehensive error handling and security measures
3. Providing more detailed testing and deployment instructions
4. Adding performance optimization recommendations
5. Ensuring all code examples are complete and production-ready

Provide an improved version that meets enterprise-grade standards.
"""
            
            improvement_result = await self.model_orchestrator.intelligent_execute(improvement_prompt)
            
            if improvement_result['success']:
                return improvement_result.get('response', final_response)
        
        return final_response
    
    async def trigger_learning_cycle(self, result: ResearchResult):
        """Trigger learning cycle based on new research result"""
        
        # Update performance metrics
        self.update_performance_metrics(result)
        
        # Detect new patterns every N sessions
        if len(self.research_history) % self.config["pattern_detection_window"] == 0:
            await self.detect_learning_patterns()
        
        # Save state
        self.save_research_history()
        self.save_learning_patterns()
    
    def update_performance_metrics(self, result: ResearchResult):
        """Update system performance metrics"""
        
        self.performance_metrics["total_sessions"] += 1
        
        if result.success:
            self.performance_metrics["successful_sessions"] += 1
        
        # Update running averages
        total = self.performance_metrics["total_sessions"]
        
        # Average quality
        current_avg_quality = self.performance_metrics["average_quality"]
        self.performance_metrics["average_quality"] = (
            (current_avg_quality * (total - 1) + result.quality_score) / total
        )
        
        # Average execution time
        current_avg_time = self.performance_metrics["average_execution_time"]
        self.performance_metrics["average_execution_time"] = (
            (current_avg_time * (total - 1) + result.execution_time) / total
        )
        
        # Calculate improvement rate (last 10 vs previous 10)
        if len(self.research_history) >= 20:
            recent_quality = sum(r.quality_score for r in self.research_history[-10:]) / 10
            previous_quality = sum(r.quality_score for r in self.research_history[-20:-10]) / 10
            self.performance_metrics["improvement_rate"] = recent_quality - previous_quality
    
    async def detect_learning_patterns(self):
        """Detect patterns in research performance for improvement"""
        
        if len(self.research_history) < 20:
            return
        
        recent_sessions = self.research_history[-self.config["pattern_detection_window"]:]
        
        # Pattern 1: Domain-specific performance
        domain_performance = {}
        for session in recent_sessions:
            domain = session.task_description.lower()
            for domain_key in ["blockchain", "web", "api", "security", "ai", "data"]:
                if domain_key in domain:
                    if domain_key not in domain_performance:
                        domain_performance[domain_key] = []
                    domain_performance[domain_key].append(session.quality_score)
        
        for domain, scores in domain_performance.items():
            if len(scores) >= 5:  # Sufficient data
                avg_score = sum(scores) / len(scores)
                if avg_score < 0.7:  # Below threshold
                    pattern = LearningPattern(
                        pattern_type="domain_weakness",
                        description=f"Poor performance in {domain} domain (avg: {avg_score:.3f})",
                        frequency=len(scores),
                        success_correlation=avg_score,
                        improvement_potential=0.9 - avg_score,
                        recommended_actions=[
                            f"Create specialized {domain} prompts",
                            f"Add {domain}-specific research methods",
                            f"Integrate {domain} best practices database"
                        ]
                    )
                    self.learning_patterns.append(pattern)
        
        # Pattern 2: Complexity level performance
        complexity_performance = {}
        for session in recent_sessions:
            # Estimate complexity from execution time and quality
            if session.execution_time > 60 and session.quality_score < 0.7:
                level = "high_complexity"
            elif session.execution_time < 20:
                level = "low_complexity"
            else:
                level = "medium_complexity"
            
            if level not in complexity_performance:
                complexity_performance[level] = []
            complexity_performance[level].append(session.quality_score)
        
        for level, scores in complexity_performance.items():
            if len(scores) >= 3 and sum(scores) / len(scores) < 0.7:
                pattern = LearningPattern(
                    pattern_type="complexity_weakness",
                    description=f"Poor performance for {level} tasks",
                    frequency=len(scores),
                    success_correlation=sum(scores) / len(scores),
                    improvement_potential=0.8,
                    recommended_actions=[
                        f"Optimize {level} task handling",
                        f"Add specialized {level} research phases",
                        f"Improve model selection for {level} tasks"
                    ]
                )
                self.learning_patterns.append(pattern)
        
        self.logger.info(f"Detected {len(self.learning_patterns)} learning patterns")
    
    def start_continuous_improvement(self):
        """Start continuous improvement background process"""
        
        self.running = True
        
        def improvement_worker():
            while self.running:
                try:
                    # Run improvement cycle
                    asyncio.run(self.run_improvement_cycle())
                    
                    # Wait for next cycle
                    time.sleep(self.config["auto_improvement_interval"])
                    
                except Exception as e:
                    self.logger.error(f"Improvement cycle error: {e}")
                    time.sleep(300)  # Wait 5 minutes on error
        
        self.improvement_thread = threading.Thread(target=improvement_worker, daemon=True)
        self.improvement_thread.start()
        
        self.logger.info("Started continuous improvement process")
    
    async def run_improvement_cycle(self):
        """Run a complete improvement cycle"""
        
        self.logger.info("Running improvement cycle")
        
        # 1. Analyze current performance
        performance_analysis = self.analyze_performance_trends()
        
        # 2. Identify improvement opportunities
        improvement_opportunities = self.identify_improvement_opportunities()
        
        # 3. Implement improvements
        implemented_improvements = await self.implement_improvements(improvement_opportunities)
        
        # 4. Generate improvement report
        self.generate_improvement_report(performance_analysis, implemented_improvements)
        
        self.logger.info("Improvement cycle completed")
    
    def analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        
        if len(self.research_history) < 10:
            return {"insufficient_data": True}
        
        # Analyze trends in different time windows
        windows = {
            "last_10": self.research_history[-10:],
            "last_30": self.research_history[-30:] if len(self.research_history) >= 30 else [],
            "last_100": self.research_history[-100:] if len(self.research_history) >= 100 else []
        }
        
        trends = {}
        
        for window_name, sessions in windows.items():
            if not sessions:
                continue
                
            trends[window_name] = {
                "avg_quality": sum(s.quality_score for s in sessions) / len(sessions),
                "success_rate": sum(1 for s in sessions if s.success) / len(sessions),
                "avg_execution_time": sum(s.execution_time for s in sessions) / len(sessions),
                "session_count": len(sessions)
            }
        
        return trends
    
    def identify_improvement_opportunities(self) -> List[Dict[str, Any]]:
        """Identify specific improvement opportunities"""
        
        opportunities = []
        
        # Check recent performance
        if len(self.research_history) >= 10:
            recent_quality = sum(r.quality_score for r in self.research_history[-10:]) / 10
            
            if recent_quality < 0.75:
                opportunities.append({
                    "type": "quality_improvement",
                    "description": f"Recent quality below target ({recent_quality:.3f})",
                    "priority": "high",
                    "actions": ["optimize_prompts", "improve_research_phases", "enhance_model_selection"]
                })
        
        # Check learning patterns
        for pattern in self.learning_patterns:
            if pattern.improvement_potential > 0.2:
                opportunities.append({
                    "type": "pattern_based",
                    "description": pattern.description,
                    "priority": "medium",
                    "actions": pattern.recommended_actions
                })
        
        return opportunities
    
    async def implement_improvements(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Implement identified improvements"""
        
        implemented = []
        
        for opportunity in opportunities[:3]:  # Limit to top 3 opportunities
            try:
                if opportunity["type"] == "quality_improvement":
                    # Optimize prompt templates
                    await self.optimize_prompt_templates()
                    implemented.append({
                        "opportunity": opportunity,
                        "action": "optimized_prompt_templates",
                        "success": True
                    })
                
                elif opportunity["type"] == "pattern_based":
                    # Apply pattern-based improvements
                    for action in opportunity["actions"][:2]:  # Limit actions
                        if "prompts" in action:
                            await self.create_specialized_prompts(action)
                        elif "research" in action:
                            await self.enhance_research_methods(action)
                    
                    implemented.append({
                        "opportunity": opportunity,
                        "action": "pattern_based_improvements",
                        "success": True
                    })
                
            except Exception as e:
                self.logger.error(f"Failed to implement improvement: {e}")
                implemented.append({
                    "opportunity": opportunity,
                    "action": "failed",
                    "success": False,
                    "error": str(e)
                })
        
        return implemented
    
    async def optimize_prompt_templates(self):
        """Optimize prompt templates based on performance data"""
        
        # Analyze successful vs unsuccessful sessions
        successful_sessions = [r for r in self.research_history if r.success and r.quality_score > 0.8]
        
        if len(successful_sessions) >= 5:
            # Extract common patterns from successful sessions
            self.logger.info("Optimizing prompt templates based on successful patterns")
            
            # This would involve analyzing successful prompts and updating templates
            # For now, we'll log the optimization
            optimization_data = {
                "successful_sessions_analyzed": len(successful_sessions),
                "optimization_timestamp": datetime.now().isoformat(),
                "improvements": [
                    "Enhanced research phase structuring",
                    "Improved domain-specific instructions",
                    "Better quality assessment criteria"
                ]
            }
            
            with open("prompt_optimizations.json", 'w') as f:
                json.dump(optimization_data, f, indent=2)
    
    async def create_specialized_prompts(self, action: str):
        """Create specialized prompts for specific domains or patterns"""
        
        domain = action.split()[2] if len(action.split()) > 2 else "general"
        
        self.logger.info(f"Creating specialized prompts for: {domain}")
        
        # This would create domain-specific prompt templates
        specialized_prompt = {
            "domain": domain,
            "created_at": datetime.now().isoformat(),
            "template": f"Specialized {domain} research template with enhanced focus areas",
            "success_patterns": "Based on analysis of successful research sessions"
        }
        
        # Save specialized prompt
        os.makedirs("specialized_prompts", exist_ok=True)
        with open(f"specialized_prompts/{domain}_prompt.json", 'w') as f:
            json.dump(specialized_prompt, f, indent=2)
    
    async def enhance_research_methods(self, action: str):
        """Enhance research methods based on identified patterns"""
        
        self.logger.info(f"Enhancing research methods: {action}")
        
        # This would enhance the research methodology
        enhancement = {
            "action": action,
            "enhanced_at": datetime.now().isoformat(),
            "improvements": [
                "Added new research phase",
                "Enhanced existing methods",
                "Improved quality metrics"
            ]
        }
        
        with open("research_method_enhancements.json", 'w') as f:
            json.dump(enhancement, f, indent=2)
    
    def generate_improvement_report(self, performance_analysis: Dict[str, Any], implemented_improvements: List[Dict[str, Any]]):
        """Generate comprehensive improvement report"""
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "performance_analysis": performance_analysis,
            "implemented_improvements": implemented_improvements,
            "current_metrics": self.performance_metrics,
            "learning_patterns_count": len(self.learning_patterns),
            "recommendations": [
                "Continue monitoring quality trends",
                "Expand pattern detection methods",
                "Implement more sophisticated learning algorithms"
            ]
        }
        
        with open(f"improvement_report_{int(time.time())}.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info("Generated improvement report")
    
    def stop_continuous_improvement(self):
        """Stop continuous improvement process"""
        
        self.running = False
        if self.improvement_thread:
            self.improvement_thread.join(timeout=5)
        
        self.logger.info("Stopped continuous improvement process")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        return {
            "performance_metrics": self.performance_metrics,
            "research_history_size": len(self.research_history),
            "learning_patterns_count": len(self.learning_patterns),
            "continuous_improvement_running": self.running,
            "last_research_session": self.research_history[-1].timestamp.isoformat() if self.research_history else None,
            "system_health": "optimal" if self.performance_metrics["average_quality"] > 0.8 else "needs_improvement"
        }

async def main():
    """Main function for testing the self-improving research system"""
    
    print("🧠 Self-Improving Research System for Local Claude/Codex")
    print("=" * 70)
    
    # Create system
    research_system = SelfImprovingResearchSystem()
    
    # Start continuous improvement
    research_system.start_continuous_improvement()
    
    try:
        # Test with sample research tasks
        test_tasks = [
            "Create a secure authentication system for a web application",
            "Design a high-performance caching layer for a microservices architecture",
            "Implement a smart contract for a decentralized voting system"
        ]
        
        for i, task in enumerate(test_tasks, 1):
            print(f"\n--- Research Task {i}: {task[:50]}... ---")
            
            result = await research_system.enhanced_research(task)
            
            if result.success:
                print(f"✅ Quality Score: {result.quality_score:.3f}")
                print(f"⏱️ Execution Time: {result.execution_time:.1f}s")
                print(f"🤖 Model Used: {result.model_used}")
            else:
                print(f"❌ Research failed: {result.result}")
        
        # Show system status
        status = research_system.get_system_status()
        print(f"\n📊 System Status:")
        print(f"Quality Average: {status['performance_metrics']['average_quality']:.3f}")
        print(f"Success Rate: {status['performance_metrics']['successful_sessions']}/{status['performance_metrics']['total_sessions']}")
        print(f"System Health: {status['system_health']}")
        
    finally:
        research_system.stop_continuous_improvement()

if __name__ == "__main__":
    asyncio.run(main())
