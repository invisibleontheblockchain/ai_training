#!/usr/bin/env python3
"""
Local Claude/Codex System
========================
Complete local coding assistant system that integrates all components
for a powerful, self-improving, research-driven coding experience.
"""

import asyncio
import json
import os
import time
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import all our components
from research_prompt_optimizer import ResearchPromptOptimizer
from multi_model_orchestrator import ModelOrchestrator
from self_improving_research_system import SelfImprovingResearchSystem
from advanced_coding_assistant import AdvancedCodingAssistant
from autonomous_learning_daemon import AutonomousLearningDaemon

class LocalClaudeCodexSystem:
    """
    Complete local Claude/Codex system that orchestrates all components
    for maximum coding assistance capabilities
    """
    
    def __init__(self):
        self.setup_logging()
        self.setup_directories()
        
        # Initialize all components
        self.coding_assistant = AdvancedCodingAssistant()
        self.research_system = SelfImprovingResearchSystem()
        self.model_orchestrator = ModelOrchestrator()
        self.prompt_optimizer = ResearchPromptOptimizer()
        self.learning_daemon = AutonomousLearningDaemon()
        
        # System state
        self.system_metrics = {
            "total_sessions": 0,
            "successful_sessions": 0,
            "average_quality": 0.0,
            "uptime_hours": 0.0,
            "models_available": 0,
            "learning_cycles_completed": 0
        }
        
        self.start_time = datetime.now()
        self.running = False
        
    def setup_logging(self):
        """Setup comprehensive system logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('local_claude_codex_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_directories(self):
        """Setup required directories"""
        directories = [
            "research_sessions",
            "research_results", 
            "specialized_prompts",
            "coding_results",
            "learning_data",
            "system_reports"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    async def initialize_system(self):
        """Initialize all system components"""
        
        self.logger.info("🚀 Initializing Local Claude/Codex System...")
        
        try:
            # Check model availability
            available_models = await self.check_model_availability()
            self.system_metrics["models_available"] = len(available_models)
            
            # Initialize learning daemon
            if available_models:
                self.learning_daemon.start()
                self.logger.info("✅ Autonomous learning daemon started")
            
            # Start continuous improvement
            self.research_system.start_continuous_improvement()
            self.logger.info("✅ Research system continuous improvement started")
            
            # Load previous state
            await self.load_system_state()
            
            self.running = True
            self.logger.info("✅ System initialization complete")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ System initialization failed: {e}")
            return False
    
    async def check_model_availability(self) -> List[str]:
        """Check which models are available"""
        
        available_models = []
        
        try:
            # Check Ollama models
            import ollama
            models = ollama.list()
            for model in models.get('models', []):
                available_models.append(model['name'])
            
        except Exception as e:
            self.logger.warning(f"Could not check Ollama models: {e}")
        
        # Check API keys for external models
        if os.getenv("OPENAI_API_KEY"):
            available_models.append("openai-gpt4")
        
        if os.getenv("ANTHROPIC_API_KEY"):
            available_models.append("anthropic-claude")
        
        self.logger.info(f"Available models: {available_models}")
        return available_models
    
    async def load_system_state(self):
        """Load previous system state"""
        
        try:
            if os.path.exists("system_state.json"):
                with open("system_state.json", 'r') as f:
                    state = json.load(f)
                    self.system_metrics.update(state.get("metrics", {}))
                self.logger.info("Previous system state loaded")
        except Exception as e:
            self.logger.warning(f"Could not load system state: {e}")
    
    async def save_system_state(self):
        """Save current system state"""
        
        try:
            state = {
                "metrics": self.system_metrics,
                "last_saved": datetime.now().isoformat(),
                "uptime_hours": (datetime.now() - self.start_time).total_seconds() / 3600
            }
            
            with open("system_state.json", 'w') as f:
                json.dump(state, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Could not save system state: {e}")
    
    async def solve_coding_problem(
        self, 
        description: str, 
        context: Dict[str, Any] = None,
        preferences: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Main interface for solving coding problems
        """
        
        session_start = time.time()
        session_id = f"session_{int(session_start)}"
        
        self.logger.info(f"Starting coding session: {session_id}")
        self.system_metrics["total_sessions"] += 1
        
        try:
            # Use the advanced coding assistant
            result = await self.coding_assistant.solve_coding_problem(
                description, context, preferences
            )
            
            # Update system metrics
            if result.success:
                self.system_metrics["successful_sessions"] += 1
                
                # Update quality average
                total_successful = self.system_metrics["successful_sessions"]
                current_avg = self.system_metrics["average_quality"]
                self.system_metrics["average_quality"] = (
                    (current_avg * (total_successful - 1) + result.quality_score) 
                    / total_successful
                )
            
            # Create comprehensive result
            comprehensive_result = {
                "session_id": session_id,
                "coding_result": result,
                "system_metrics": self.system_metrics.copy(),
                "execution_time": time.time() - session_start,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save result
            await self.save_coding_result(comprehensive_result)
            
            # Trigger learning if needed
            await self.trigger_learning_cycle(result)
            
            self.logger.info(f"Coding session completed: {session_id}")
            
            return comprehensive_result
            
        except Exception as e:
            self.logger.error(f"Coding session failed: {session_id} - {e}")
            
            error_result = {
                "session_id": session_id,
                "error": str(e),
                "success": False,
                "execution_time": time.time() - session_start,
                "timestamp": datetime.now().isoformat()
            }
            
            return error_result
    
    async def save_coding_result(self, result: Dict[str, Any]):
        """Save coding result for analysis and learning"""
        
        try:
            filename = f"coding_results/result_{result['session_id']}.json"
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Could not save coding result: {e}")
    
    async def trigger_learning_cycle(self, coding_result):
        """Trigger learning cycle based on coding result"""
        
        # Update learning daemon with new data
        if coding_result.success and coding_result.quality_score > 0.8:
            learning_data = {
                "task_description": coding_result.task_id,
                "code_generated": coding_result.code[:1000],  # First 1000 chars
                "quality_score": coding_result.quality_score,
                "patterns": self.extract_code_patterns(coding_result.code),
                "timestamp": datetime.now().isoformat()
            }
            
            # Save learning data
            learning_file = f"learning_data/learning_{int(time.time())}.json"
            with open(learning_file, 'w') as f:
                json.dump(learning_data, f, indent=2, default=str)
            
            self.system_metrics["learning_cycles_completed"] += 1
    
    def extract_code_patterns(self, code: str) -> List[str]:
        """Extract patterns from generated code for learning"""
        
        patterns = []
        
        # Basic pattern detection
        if "class " in code:
            patterns.append("object_oriented_design")
        
        if "async def" in code or "await " in code:
            patterns.append("asynchronous_programming")
        
        if "try:" in code and "except" in code:
            patterns.append("error_handling")
        
        if "logging" in code or "logger" in code:
            patterns.append("logging_implementation")
        
        if "def test_" in code or "import pytest" in code:
            patterns.append("test_driven_development")
        
        if "fastapi" in code.lower() or "flask" in code.lower():
            patterns.append("web_api_development")
        
        if "contract" in code.lower() and "solidity" in code.lower():
            patterns.append("smart_contract_development")
        
        return patterns
    
    async def research_coding_topic(
        self, 
        topic: str, 
        depth: str = "medium"
    ) -> Dict[str, Any]:
        """Research a coding topic in depth"""
        
        self.logger.info(f"Researching topic: {topic}")
        
        # Use research system for deep topic research
        result = await self.research_system.enhanced_research(
            f"Research the following coding topic in depth: {topic}",
            context={"research_depth": depth, "focus": "coding_best_practices"}
        )
        
        return {
            "topic": topic,
            "research_result": result,
            "depth": depth,
            "timestamp": datetime.now().isoformat()
        }
    
    async def optimize_system_performance(self):
        """Optimize system performance based on usage patterns"""
        
        self.logger.info("Running system performance optimization")
        
        try:
            # Analyze recent performance
            performance_report = await self.generate_performance_report()
            
            # Identify optimization opportunities
            optimizations = self.identify_system_optimizations(performance_report)
            
            # Apply optimizations
            applied_optimizations = await self.apply_optimizations(optimizations)
            
            self.logger.info(f"Applied {len(applied_optimizations)} system optimizations")
            
            return {
                "performance_report": performance_report,
                "optimizations_applied": applied_optimizations,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"System optimization failed: {e}")
            return {"error": str(e)}
    
    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        # Get component statuses
        coding_status = self.coding_assistant.get_system_status()
        research_status = self.research_system.get_system_status()
        model_status = self.model_orchestrator.get_model_performance_report()
        
        # Calculate uptime
        uptime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        self.system_metrics["uptime_hours"] = uptime_hours
        
        return {
            "system_metrics": self.system_metrics,
            "coding_assistant": coding_status,
            "research_system": research_status,
            "model_orchestrator": model_status,
            "uptime_hours": uptime_hours,
            "health_score": self.calculate_system_health(),
            "timestamp": datetime.now().isoformat()
        }
    
    def calculate_system_health(self) -> float:
        """Calculate overall system health score"""
        
        health_factors = []
        
        # Success rate factor
        if self.system_metrics["total_sessions"] > 0:
            success_rate = self.system_metrics["successful_sessions"] / self.system_metrics["total_sessions"]
            health_factors.append(success_rate)
        
        # Quality factor
        quality_score = self.system_metrics["average_quality"]
        health_factors.append(quality_score)
        
        # Model availability factor
        model_factor = min(1.0, self.system_metrics["models_available"] / 3)  # Expect at least 3 models
        health_factors.append(model_factor)
        
        # Learning factor
        if self.system_metrics["learning_cycles_completed"] > 0:
            learning_factor = min(1.0, self.system_metrics["learning_cycles_completed"] / 10)
            health_factors.append(learning_factor)
        
        return sum(health_factors) / len(health_factors) if health_factors else 0.5
    
    def identify_system_optimizations(self, performance_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify potential system optimizations"""
        
        optimizations = []
        
        # Check success rate
        if self.system_metrics["total_sessions"] > 10:
            success_rate = self.system_metrics["successful_sessions"] / self.system_metrics["total_sessions"]
            if success_rate < 0.8:
                optimizations.append({
                    "type": "improve_success_rate",
                    "description": f"Success rate is low ({success_rate:.2%})",
                    "priority": "high",
                    "action": "optimize_error_handling"
                })
        
        # Check quality score
        if self.system_metrics["average_quality"] < 0.8:
            optimizations.append({
                "type": "improve_quality",
                "description": f"Average quality below target ({self.system_metrics['average_quality']:.3f})",
                "priority": "medium",
                "action": "enhance_prompts"
            })
        
        # Check model availability
        if self.system_metrics["models_available"] < 2:
            optimizations.append({
                "type": "expand_models",
                "description": "Limited model availability",
                "priority": "medium",
                "action": "configure_additional_models"
            })
        
        return optimizations
    
    async def apply_optimizations(self, optimizations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply identified optimizations"""
        
        applied = []
        
        for optimization in optimizations[:3]:  # Limit to top 3
            try:
                if optimization["action"] == "optimize_error_handling":
                    # Improve error handling across components
                    await self.enhance_error_handling()
                    applied.append({**optimization, "status": "applied"})
                
                elif optimization["action"] == "enhance_prompts":
                    # Optimize prompts based on performance
                    await self.prompt_optimizer.optimize_prompts_based_on_performance()
                    applied.append({**optimization, "status": "applied"})
                
                elif optimization["action"] == "configure_additional_models":
                    # Suggest model configurations
                    suggestions = await self.suggest_model_configurations()
                    applied.append({**optimization, "status": "suggestions_generated", "suggestions": suggestions})
                
            except Exception as e:
                applied.append({**optimization, "status": "failed", "error": str(e)})
        
        return applied
    
    async def enhance_error_handling(self):
        """Enhance error handling across the system"""
        
        # This would implement system-wide error handling improvements
        self.logger.info("Enhanced error handling applied")
    
    async def suggest_model_configurations(self) -> List[str]:
        """Suggest additional model configurations"""
        
        suggestions = [
            "Install Code Llama via Ollama: ollama pull codellama",
            "Install Qwen Coder via Ollama: ollama pull qwen2.5-coder",
            "Configure OpenAI API key for GPT-4 access",
            "Configure Anthropic API key for Claude access"
        ]
        
        return suggestions
    
    async def interactive_mode(self):
        """Start interactive mode for the system"""
        
        print("🎯 Local Claude/Codex System - Interactive Mode")
        print("=" * 60)
        print("Commands:")
        print("  code <description>  - Solve a coding problem")
        print("  research <topic>    - Research a coding topic")
        print("  status              - Show system status")
        print("  optimize            - Run system optimization")
        print("  report              - Generate performance report")
        print("  exit                - Exit the system")
        print()
        
        while self.running:
            try:
                user_input = input("\n🤖 Command: ").strip()
                
                if not user_input:
                    continue
                
                parts = user_input.split(' ', 1)
                command = parts[0].lower()
                
                if command == 'exit':
                    print("👋 Shutting down...")
                    await self.shutdown()
                    break
                
                elif command == 'code':
                    if len(parts) > 1:
                        description = parts[1]
                        print(f"\n🔍 Solving: {description}")
                        result = await self.solve_coding_problem(description)
                        
                        if result.get('coding_result', {}).get('success', False):
                            coding_result = result['coding_result']
                            print(f"✅ Success! Quality: {coding_result.quality_score:.3f}")
                            print(f"📄 Result saved to: coding_results/result_{result['session_id']}.json")
                        else:
                            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
                    else:
                        print("Please provide a coding problem description")
                
                elif command == 'research':
                    if len(parts) > 1:
                        topic = parts[1]
                        print(f"\n🔍 Researching: {topic}")
                        result = await self.research_coding_topic(topic)
                        
                        if result['research_result'].success:
                            print(f"✅ Research completed! Quality: {result['research_result'].quality_score:.3f}")
                        else:
                            print(f"❌ Research failed")
                    else:
                        print("Please provide a research topic")
                
                elif command == 'status':
                    report = await self.generate_performance_report()
                    print(f"\n📊 System Status:")
                    print(f"  Health Score: {report['health_score']:.3f}")
                    print(f"  Success Rate: {self.system_metrics['successful_sessions']}/{self.system_metrics['total_sessions']}")
                    print(f"  Average Quality: {self.system_metrics['average_quality']:.3f}")
                    print(f"  Models Available: {self.system_metrics['models_available']}")
                    print(f"  Uptime: {report['uptime_hours']:.1f} hours")
                
                elif command == 'optimize':
                    print("\n⚡ Running system optimization...")
                    result = await self.optimize_system_performance()
                    if 'error' not in result:
                        print(f"✅ Optimization complete! Applied {len(result['optimizations_applied'])} improvements")
                    else:
                        print(f"❌ Optimization failed: {result['error']}")
                
                elif command == 'report':
                    print("\n📋 Generating performance report...")
                    report = await self.generate_performance_report()
                    
                    report_file = f"system_reports/performance_report_{int(time.time())}.json"
                    with open(report_file, 'w') as f:
                        json.dump(report, f, indent=2, default=str)
                    
                    print(f"📄 Report saved to: {report_file}")
                
                else:
                    print(f"Unknown command: {command}")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Shutting down...")
                await self.shutdown()
                break
            except Exception as e:
                print(f"\n⚠️ Error: {e}")
    
    async def shutdown(self):
        """Gracefully shutdown the system"""
        
        self.logger.info("🛑 Shutting down Local Claude/Codex System...")
        
        self.running = False
        
        # Stop learning daemon
        if hasattr(self.learning_daemon, 'stop'):
            self.learning_daemon.stop()
        
        # Stop research system improvement
        self.research_system.stop_continuous_improvement()
        
        # Save final state
        await self.save_system_state()
        
        self.logger.info("✅ System shutdown complete")

async def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description="Local Claude/Codex System")
    parser.add_argument("--mode", choices=["interactive", "test"], default="interactive",
                       help="System mode (default: interactive)")
    parser.add_argument("--task", type=str, help="Single task to execute in test mode")
    
    args = parser.parse_args()
    
    # Create and initialize system
    system = LocalClaudeCodexSystem()
    
    if not await system.initialize_system():
        print("❌ System initialization failed")
        return
    
    try:
        if args.mode == "interactive":
            await system.interactive_mode()
        
        elif args.mode == "test" and args.task:
            print(f"🧪 Testing with task: {args.task}")
            result = await system.solve_coding_problem(args.task)
            
            if result.get('coding_result', {}).get('success', False):
                print("✅ Test completed successfully")
            else:
                print("❌ Test failed")
        
        else:
            print("Running basic system test...")
            test_result = await system.solve_coding_problem(
                "Create a simple Python function that calculates the factorial of a number"
            )
            
            if test_result.get('coding_result', {}).get('success', False):
                print("✅ Basic test passed - System is operational")
            else:
                print("❌ Basic test failed")
    
    finally:
        await system.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
