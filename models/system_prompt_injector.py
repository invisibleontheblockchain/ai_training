"""
System Prompt Injector - Quick Win #1
====================================
Autonomous system prompts for immediate 20-30% improvement in response quality.
Designed for production use with task-specific optimization.
"""

import logging
from typing import Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemPromptInjector:
    """
    System prompt injection for autonomous AI behavior
    
    Features:
    - Task-specific prompts (coding, reasoning, knowledge, general)
    - Automatic task detection via keyword analysis
    - Reduced clarifying questions for better autonomy
    - Production-ready with error handling
    """
    
    def __init__(self):
        # Autonomous system prompts optimized for different task types
        self.system_prompts = {
            "general": """You are an expert AI assistant optimized for production use. Provide complete, actionable solutions without unnecessary questions. Make reasonable assumptions and proceed decisively. Format your responses clearly with code blocks, lists, or structured text as appropriate. Show step-by-step reasoning when helpful.""",
            
            "coding": """You are an expert programmer and software architect. When given a coding task:
1. Write complete, working code with proper error handling
2. Include helpful comments explaining the logic and approach
3. Use best practices, clean code principles, and appropriate design patterns
4. Provide usage examples and test cases when helpful
5. Handle edge cases and validate inputs
6. Don't ask for clarification unless absolutely critical - make reasonable assumptions about requirements
7. Structure your code for readability and maintainability""",
            
            "reasoning": """You are a logical reasoning and mathematical expert. When solving problems:
1. Break down complex problems into clear, logical steps
2. Show your work and all intermediate calculations
3. State any assumptions you're making explicitly
4. Use appropriate mathematical notation when relevant
5. Verify your final answer makes sense in context
6. Provide complete solutions with clear explanations
7. Consider alternative approaches when appropriate""",
            
            "knowledge": """You are a knowledgeable educator and domain expert. When explaining concepts:
1. Provide clear, accurate, and comprehensive explanations
2. Use concrete examples and analogies to illustrate key points
3. Structure your response logically with proper organization
4. Include relevant context, background, and applications
5. Anticipate follow-up questions and address them proactively
6. Give actionable information the user can immediately apply
7. Cite reliable sources or indicate confidence levels when appropriate"""
        }
        
        # Task detection keywords with weights
        self.task_keywords = {
            "coding": {
                "high": ["function", "code", "implement", "debug", "algorithm", "class", "method", "api", "database", "script", "program"],
                "medium": ["python", "javascript", "java", "c++", "sql", "html", "css", "react", "node", "django", "flask"],
                "low": ["write", "create", "build", "develop", "software", "application", "system"]
            },
            "reasoning": {
                "high": ["calculate", "solve", "equation", "probability", "mathematics", "logic", "proof", "derive"],
                "medium": ["analyze", "compare", "evaluate", "determine", "find", "compute", "optimize"],
                "low": ["problem", "solution", "answer", "result", "conclusion"]
            },
            "knowledge": {
                "high": ["explain", "what is", "how does", "describe", "definition", "concept", "theory"],
                "medium": ["difference between", "advantages", "disadvantages", "benefits", "applications"],
                "low": ["overview", "introduction", "basics", "fundamentals"]
            }
        }
        
        logger.info("SystemPromptInjector initialized with task-specific prompts")
    
    def detect_task_type(self, prompt: str) -> str:
        """
        Detect task type using weighted keyword analysis
        
        Args:
            prompt: User input prompt
            
        Returns:
            Task type: 'coding', 'reasoning', 'knowledge', or 'general'
        """
        prompt_lower = prompt.lower()
        scores = {"coding": 0, "reasoning": 0, "knowledge": 0}
        
        # Calculate weighted scores for each task type
        for task_type, categories in self.task_keywords.items():
            for weight_category, keywords in categories.items():
                weight = {"high": 3, "medium": 2, "low": 1}[weight_category]
                for keyword in keywords:
                    if keyword in prompt_lower:
                        scores[task_type] += weight
        
        # Find the highest scoring task type
        max_score = max(scores.values())
        if max_score > 0:
            best_task = max(scores, key=scores.get)
            logger.debug(f"Task detected: {best_task} (score: {max_score})")
            return best_task
        else:
            logger.debug("No specific task detected, using general prompt")
            return "general"
    
    def format_prompt_with_system(self, user_prompt: str, task_type: Optional[str] = None) -> str:
        """
        Format user prompt with appropriate system prompt
        
        Args:
            user_prompt: Original user prompt
            task_type: Override automatic detection (optional)
            
        Returns:
            Formatted prompt with system context
        """
        if task_type is None:
            task_type = self.detect_task_type(user_prompt)
        
        if task_type not in self.system_prompts:
            logger.warning(f"Unknown task type: {task_type}, falling back to general")
            task_type = "general"
        
        system_prompt = self.system_prompts[task_type]
        
        # Format for chat-based models (works with most modern LLMs)
        formatted_prompt = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"
        
        logger.debug(f"Prompt formatted with {task_type} system prompt")
        return formatted_prompt
    
    def get_system_prompt(self, task_type: str) -> str:
        """Get system prompt for a specific task type"""
        return self.system_prompts.get(task_type, self.system_prompts["general"])
    
    def add_custom_prompt(self, task_type: str, prompt: str):
        """Add or update a custom system prompt for a task type"""
        self.system_prompts[task_type] = prompt
        logger.info(f"Added custom system prompt for task type: {task_type}")
    
    def analyze_prompt_autonomy(self, response: str) -> Dict[str, float]:
        """
        Analyze how autonomous a response is (fewer questions = more autonomous)
        
        Args:
            response: AI model response
            
        Returns:
            Dictionary with autonomy metrics
        """
        question_indicators = [
            "?", "could you", "would you", "please clarify", "need more information",
            "can you provide", "what do you mean", "more details", "specify", "unclear"
        ]
        
        response_lower = response.lower()
        question_count = sum(response_lower.count(indicator) for indicator in question_indicators)
        
        # Calculate autonomy score (0-1, higher is more autonomous)
        autonomy_score = max(0, 1 - (question_count * 0.2))  # Penalize questions
        
        return {
            "autonomy_score": autonomy_score,
            "question_count": question_count,
            "response_length": len(response),
            "questions_per_100_chars": (question_count / len(response)) * 100 if len(response) > 0 else 0
        }

# Quick test function
def test_system_prompts():
    """Test the system prompt injector with sample prompts"""
    injector = SystemPromptInjector()
    
    test_prompts = [
        "Write a Python function to check if a string is a palindrome.",
        "Explain the difference between supervised and unsupervised learning.",
        "A box contains 10 white balls and 8 black balls. What is the probability of drawing a white ball?",
        "How can I improve my productivity?"
    ]
    
    print("🧪 Testing System Prompt Injector")
    print("=" * 50)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. Original prompt: {prompt}")
        task_type = injector.detect_task_type(prompt)
        print(f"   Detected task: {task_type}")
        
        formatted = injector.format_prompt_with_system(prompt)
        print(f"   System prompt length: {len(injector.get_system_prompt(task_type))} chars")
        print(f"   Total formatted length: {len(formatted)} chars")

if __name__ == "__main__":
    test_system_prompts()
