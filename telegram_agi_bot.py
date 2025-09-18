#!/usr/bin/env python3
"""
Telegram AGI Bot Interface
==========================
Telegram bot interface for controlling the Master AGI System.
Allows natural conversation with AGI agents and real-time agent management.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, filters
)

# Import our AGI system
from master_agi_system import MasterAGISystem, get_master_agi

class TelegramAGIBot:
    """Telegram interface for AGI agent swarm"""
    
    def __init__(self, token: str):
        self.token = token
        self.agi_system = get_master_agi()
        self.user_conversations: Dict[int, Dict] = {}  # Track user conversations
        self.active_agents: Dict[int, str] = {}  # Track which agent user is talking to
        self.setup_logging()
        
        # Initialize application
        self.app = Application.builder().token(token).build()
        self.setup_handlers()
    
    def setup_logging(self):
        """Setup logging for Telegram bot"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('telegram_agi_bot.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def setup_handlers(self):
        """Setup Telegram bot handlers"""
        # Command handlers
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(CommandHandler("help", self.help_command))
        self.app.add_handler(CommandHandler("spawn", self.spawn_agent_command))
        self.app.add_handler(CommandHandler("list", self.list_agents_command))
        self.app.add_handler(CommandHandler("talk", self.talk_agent_command))
        self.app.add_handler(CommandHandler("task", self.task_agent_command))
        self.app.add_handler(CommandHandler("status", self.status_command))
        self.app.add_handler(CommandHandler("swarm", self.swarm_command))
        self.app.add_handler(CommandHandler("test", self.test_agent_command))
        self.app.add_handler(CommandHandler("compete", self.compete_command))
        self.app.add_handler(CommandHandler("system", self.system_status_command))
        self.app.add_handler(CommandHandler("stop_talk", self.stop_talk_command))
        
        # Message handler for conversations
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # Callback query handler for inline keyboards
        self.app.add_handler(CallbackQueryHandler(self.handle_callback))
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user_id = update.effective_user.id
        welcome_message = """
🧠 **Welcome to the AGI Agent Ecosystem!**

I'm your interface to a self-replicating AI agent system. Here's what you can do:

**Agent Management:**
/spawn <type> <purpose> - Create new specialized agent
/list - Show all active agents
/talk <agent_id> - Start conversation with agent
/stop_talk - Stop current conversation
/status <agent_id> - Get agent performance

**Agent Operations:**
/task <agent_id> <task> - Assign task to agent
/swarm <topic> - Deploy agent swarm for research
/test <agent_id> - Test agent performance
/compete <agent1> <agent2> <task> - Agent competition

**System:**
/system - System status and metrics
/help - Show detailed help

**Agent Types:** researcher, coder, analyst, tester

Ready to create some AGI magic? 🚀
        """
        
        await update.message.reply_text(welcome_message, parse_mode='Markdown')
        
        # Initialize user conversation tracking
        self.user_conversations[user_id] = {
            "started_at": datetime.now(),
            "message_count": 0,
            "active_agent": None
        }
        
        self.logger.info(f"User {user_id} started conversation")
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = """
🔧 **AGI Agent Commands Guide**

**Creating Agents:**
`/spawn researcher "AI trends analysis"` - Create research agent
`/spawn coder "Python optimization"` - Create coding agent
`/spawn analyst "Market analysis"` - Create analysis agent
`/spawn tester "Security testing"` - Create testing agent

**Agent Interaction:**
`/talk agent_id` - Start conversation with specific agent
`/task agent_id "analyze blockchain trends"` - Assign specific task
`/test agent_id` - Test agent with sample queries

**Advanced Operations:**
`/swarm "DeFi research"` - Create specialized agent team
`/compete agent1 agent2 "code review task"` - Agent vs agent

**Examples:**
• `/spawn researcher "Monitor AI papers on arXiv"`
• `/talk researcher_a1b2c3d4`
• `/swarm "Create yield farming strategy"`
• `/compete coder_1 coder_2 "optimize smart contract"`

**Tips:**
- Agents learn from every interaction
- Use specific purposes for better performance
- Swarms are perfect for complex multi-step research
        """
        
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def spawn_agent_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /spawn command to create new agent"""
        if len(context.args) < 2:
            await update.message.reply_text(
                "❌ Usage: `/spawn <agent_type> <purpose>`\n"
                "Example: `/spawn researcher \"AI trends analysis\"`\n"
                "Agent types: researcher, coder, analyst, tester",
                parse_mode='Markdown'
            )
            return
        
        agent_type = context.args[0].lower()
        purpose = " ".join(context.args[1:]).strip('"')
        
        # Validate agent type
        valid_types = ["researcher", "coder", "analyst", "tester"]
        if agent_type not in valid_types:
            await update.message.reply_text(
                f"❌ Invalid agent type: {agent_type}\n"
                f"Valid types: {', '.join(valid_types)}"
            )
            return
        
        try:
            # Show creation progress
            progress_msg = await update.message.reply_text(
                f"🤖 Creating {agent_type} agent...\n"
                f"Purpose: {purpose}\n"
                f"⏳ Designing architecture..."
            )
            
            # Create agent
            agent_id = await self.agi_system.create_agent(agent_type, purpose)
            
            # Update progress
            await progress_msg.edit_text(
                f"✅ **Agent Created Successfully!**\n\n"
                f"🤖 **Agent ID:** `{agent_id}`\n"
                f"📝 **Type:** {agent_type.title()}\n"
                f"🎯 **Purpose:** {purpose}\n\n"
                f"Ready for tasks! Use `/talk {agent_id}` to start conversation.",
                parse_mode='Markdown'
            )
            
            # Create inline keyboard for quick actions
            keyboard = [
                [
                    InlineKeyboardButton("💬 Talk to Agent", callback_data=f"talk_{agent_id}"),
                    InlineKeyboardButton("📊 Agent Status", callback_data=f"status_{agent_id}")
                ],
                [
                    InlineKeyboardButton("🧪 Test Agent", callback_data=f"test_{agent_id}"),
                    InlineKeyboardButton("📋 List All", callback_data="list_agents")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                f"🎉 {agent_type.title()} agent is ready!",
                reply_markup=reply_markup
            )
            
            self.logger.info(f"User {update.effective_user.id} created agent {agent_id}")
            
        except Exception as e:
            await update.message.reply_text(f"❌ Failed to create agent: {str(e)}")
            self.logger.error(f"Agent creation failed: {e}")
    
    async def list_agents_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /list command to show all agents"""
        try:
            agents = self.agi_system.get_active_agents()
            
            if not agents:
                await update.message.reply_text(
                    "🤖 No agents created yet.\n"
                    "Use `/spawn <type> <purpose>` to create your first agent!",
                    parse_mode='Markdown'
                )
                return
            
            # Create agent list message
            message = "🤖 **Active AGI Agents:**\n\n"
            
            for i, agent in enumerate(agents, 1):
                status_emoji = "🟢" if agent["status"] == "active" else "🟡"
                score_emoji = "🏆" if agent["score"] > 80 else "📊" if agent["score"] > 60 else "📉"
                
                message += f"{status_emoji} **{agent['name']}**\n"
                message += f"   ID: `{agent['agent_id']}`\n"
                message += f"   Type: {agent['type'].title()}\n"
                message += f"   {score_emoji} Score: {agent['score']:.1f}/100\n"
                message += f"   📈 Tasks: {agent['tasks_completed']}\n"
                message += f"   Purpose: {agent['purpose'][:50]}...\n\n"
            
            # Add system stats
            system_status = self.agi_system.get_system_status()
            message += f"📊 **System Stats:**\n"
            message += f"Active: {system_status['active_agents']}/{system_status['total_agents']} agents\n"
            message += f"Average Performance: {system_status['avg_performance']:.1f}%\n"
            message += f"Total Tasks: {system_status['total_tasks_completed']}"
            
            # Create keyboard with agent actions
            keyboard = []
            for agent in agents[:10]:  # Limit to 10 for keyboard size
                keyboard.append([
                    InlineKeyboardButton(
                        f"💬 {agent['name'][:15]}", 
                        callback_data=f"talk_{agent['agent_id']}"
                    ),
                    InlineKeyboardButton(
                        f"📊 Status", 
                        callback_data=f"status_{agent['agent_id']}"
                    )
                ])
            
            keyboard.append([
                InlineKeyboardButton("🔄 Refresh", callback_data="list_agents"),
                InlineKeyboardButton("🆕 Create Agent", callback_data="help_spawn")
            ])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(message, parse_mode='Markdown', reply_markup=reply_markup)
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error listing agents: {str(e)}")
            self.logger.error(f"Error listing agents: {e}")
    
    async def talk_agent_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /talk command to start conversation with agent"""
        if not context.args:
            await update.message.reply_text(
                "❌ Usage: `/talk <agent_id>`\n"
                "Use `/list` to see available agents.",
                parse_mode='Markdown'
            )
            return
        
        agent_id = context.args[0]
        user_id = update.effective_user.id
        
        try:
            # Verify agent exists
            agent = self.agi_system.agent_registry.get_agent(agent_id)
            if not agent:
                await update.message.reply_text(f"❌ Agent `{agent_id}` not found. Use `/list` to see available agents.")
                return
            
            # Set active agent for user
            self.active_agents[user_id] = agent_id
            
            # Initialize conversation if needed
            if user_id not in self.user_conversations:
                self.user_conversations[user_id] = {
                    "started_at": datetime.now(),
                    "message_count": 0,
                    "active_agent": agent_id
                }
            else:
                self.user_conversations[user_id]["active_agent"] = agent_id
            
            await update.message.reply_text(
                f"💬 **Now talking to {agent.name}**\n\n"
                f"🤖 Type: {agent.agent_type.title()}\n"
                f"🎯 Purpose: {agent.purpose}\n"
                f"🏆 Score: {agent.score:.1f}/100\n\n"
                f"Just type your messages normally. Use `/stop_talk` to end conversation.\n\n"
                f"👋 **{agent.name}:** Hello! I'm ready to help with {agent.agent_type} tasks. What would you like me to work on?",
                parse_mode='Markdown'
            )
            
            self.logger.info(f"User {user_id} started talking to agent {agent_id}")
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error starting conversation: {str(e)}")
            self.logger.error(f"Error in talk command: {e}")
    
    async def stop_talk_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop_talk command"""
        user_id = update.effective_user.id
        
        if user_id in self.active_agents:
            agent_id = self.active_agents[user_id]
            agent = self.agi_system.agent_registry.get_agent(agent_id)
            agent_name = agent.name if agent else "Unknown Agent"
            
            del self.active_agents[user_id]
            
            await update.message.reply_text(
                f"👋 Conversation with **{agent_name}** ended.\n"
                f"Use `/talk <agent_id>` to start a new conversation or `/list` to see all agents.",
                parse_mode='Markdown'
            )
        else:
            await update.message.reply_text(
                "❌ You're not currently in a conversation with any agent.\n"
                "Use `/talk <agent_id>` to start a conversation."
            )
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular messages (conversations with agents)"""
        user_id = update.effective_user.id
        message_text = update.message.text
        
        # Check if user is talking to an agent
        if user_id not in self.active_agents:
            await update.message.reply_text(
                "🤖 Hi! I'm the AGI Agent Controller.\n\n"
                "To talk to an agent, use:\n"
                "• `/spawn <type> <purpose>` to create an agent\n"
                "• `/list` to see existing agents\n"
                "• `/talk <agent_id>` to start conversation\n\n"
                "Or use `/help` for all commands!"
            )
            return
        
        agent_id = self.active_agents[user_id]
        
        try:
            # Get agent info
            agent = self.agi_system.agent_registry.get_agent(agent_id)
            if not agent:
                await update.message.reply_text(
                    f"❌ Agent {agent_id} no longer exists. Use `/list` to see available agents."
                )
                del self.active_agents[user_id]
                return
            
            # Show typing indicator
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
            
            # Simulate agent processing (in real implementation, this would call actual agent)
            start_time = time.time()
            
            # Generate agent response based on type and message
            response = await self.generate_agent_response(agent, message_text)
            
            execution_time = time.time() - start_time
            
            # Update conversation tracking
            if user_id in self.user_conversations:
                self.user_conversations[user_id]["message_count"] += 1
            
            # Send response
            await update.message.reply_text(
                f"🤖 **{agent.name}:** {response}\n\n"
                f"⏱️ Response time: {execution_time:.2f}s",
                parse_mode='Markdown'
            )
            
            # Create quick action keyboard
            keyboard = [
                [
                    InlineKeyboardButton("🔄 Ask Again", callback_data=f"ask_again_{agent_id}"),
                    InlineKeyboardButton("📊 Agent Status", callback_data=f"status_{agent_id}")
                ],
                [
                    InlineKeyboardButton("🏁 End Chat", callback_data="end_chat"),
                    InlineKeyboardButton("📋 Switch Agent", callback_data="list_agents")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                "What else can I help you with?",
                reply_markup=reply_markup
            )
            
            self.logger.info(f"Agent {agent_id} responded to user {user_id} in {execution_time:.2f}s")
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error processing message: {str(e)}")
            self.logger.error(f"Error in message handling: {e}")
    
    async def generate_agent_response(self, agent, message_text: str) -> str:
        """Generate agent response based on type and specialization"""
        # Simulate different response styles based on agent type
        responses = {
            "researcher": f"I've analyzed your request '{message_text}'. Based on my {', '.join(agent.specializations)} expertise, here are my findings: [Comprehensive research results with data, trends, and insights]. I've also identified 3 key patterns and can provide detailed citations if needed.",
            
            "coder": f"I'll implement a solution for '{message_text}'. As a {', '.join(agent.specializations)} specialist, I recommend: [Clean, optimized code with proper documentation, error handling, and best practices]. The implementation includes security considerations and performance optimizations.",
            
            "analyst": f"Analyzing '{message_text}' from a {', '.join(agent.specializations)} perspective: [Statistical analysis with patterns, predictions, and risk assessment]. I've identified key metrics and can provide detailed forecasts with confidence intervals.",
            
            "tester": f"Testing approach for '{message_text}': [Comprehensive test strategy including {', '.join(agent.specializations)} testing, quality metrics, and validation procedures]. I'll ensure 95%+ coverage with performance benchmarks."
        }
        
        base_response = responses.get(agent.agent_type, f"Processing '{message_text}' with my specialized capabilities...")
        
        # Add personalization based on agent purpose
        personalized_response = f"{base_response}\n\nAs an agent specialized in '{agent.purpose}', I can also provide additional insights specific to this domain. Would you like me to dive deeper into any particular aspect?"
        
        return personalized_response
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command for agent performance"""
        if not context.args:
            await update.message.reply_text(
                "❌ Usage: `/status <agent_id>`\n"
                "Use `/list` to see available agents.",
                parse_mode='Markdown'
            )
            return
        
        agent_id = context.args[0]
        
        try:
            agent = self.agi_system.agent_registry.get_agent(agent_id)
            if not agent:
                await update.message.reply_text(f"❌ Agent `{agent_id}` not found.")
                return
            
            # Get performance metrics
            performance = self.agi_system.agent_registry.get_agent_performance(agent_id)
            
            # Calculate status indicators
            status_emoji = "🟢" if agent.score > 80 else "🟡" if agent.score > 60 else "🔴"
            trend_emoji = "📈" if len(agent.performance_history) > 1 and agent.performance_history[-1] > agent.performance_history[-2] else "📊"
            
            status_message = f"""
{status_emoji} **Agent Status Report**

🤖 **{agent.name}**
🆔 ID: `{agent.agent_id}`
📝 Type: {agent.agent_type.title()}
🎯 Purpose: {agent.purpose}

📊 **Performance Metrics:**
{trend_emoji} Score: {agent.score:.1f}/100
✅ Tasks Completed: {agent.tasks_completed}
⚡ Specializations: {', '.join(agent.specializations)}
🕐 Created: {agent.created_at.strftime('%Y-%m-%d %H:%M')}

🏆 **Capabilities:**
{chr(10).join(f"• {cap.replace('_', ' ').title()}" for cap in agent.capabilities)}

📈 **Recent Performance:**
{' '.join('🟢' if score > 0.8 else '🟡' if score > 0.6 else '🔴' for score in agent.performance_history[-10:])}
            """
            
            # Create action keyboard
            keyboard = [
                [
                    InlineKeyboardButton("💬 Talk to Agent", callback_data=f"talk_{agent_id}"),
                    InlineKeyboardButton("🧪 Test Agent", callback_data=f"test_{agent_id}")
                ],
                [
                    InlineKeyboardButton("📋 All Agents", callback_data="list_agents"),
                    InlineKeyboardButton("🔄 Refresh", callback_data=f"status_{agent_id}")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                status_message,
                parse_mode='Markdown',
                reply_markup=reply_markup
            )
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error getting status: {str(e)}")
            self.logger.error(f"Error in status command: {e}")
    
    async def swarm_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /swarm command to create agent swarm"""
        if not context.args:
            await update.message.reply_text(
                "❌ Usage: `/swarm <research_topic>`\n"
                "Example: `/swarm \"DeFi yield farming analysis\"`",
                parse_mode='Markdown'
            )
            return
        
        research_topic = " ".join(context.args).strip('"')
        
        try:
            progress_msg = await update.message.reply_text(
                f"🚀 **Deploying Agent Swarm**\n\n"
                f"📋 Topic: {research_topic}\n"
                f"⏳ Creating specialized agents..."
            )
            
            # Create agent swarm
            swarm_agents = await self.agi_system.spawn_agent_swarm(research_topic)
            
            await progress_msg.edit_text(
                f"✅ **Agent Swarm Deployed!**\n\n"
                f"📋 **Topic:** {research_topic}\n"
                f"🤖 **Agents Created:** {len(swarm_agents)}\n\n"
                f"**Swarm Composition:**\n"
                f"🔬 Research Agent - Trend analysis\n"
                f"📊 Analysis Agent - Data patterns\n"
                f"💻 Coding Agent - Implementation\n"
                f"🧪 Testing Agent - Validation\n\n"
                f"The swarm is now working together on your topic!",
                parse_mode='Markdown'
            )
            
            # Create swarm management keyboard
            keyboard = [
                [
                    InlineKeyboardButton("📋 View Swarm", callback_data="list_agents"),
                    InlineKeyboardButton("🎯 Assign Task", callback_data=f"swarm_task_{research_topic}")
                ],
                [
                    InlineKeyboardButton("📊 Swarm Status", callback_data="system_status"),
                    InlineKeyboardButton("🧪 Test Swarm", callback_data=f"test_swarm_{research_topic}")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                f"🎉 Swarm ready for {research_topic}!",
                reply_markup=reply_markup
            )
            
            self.logger.info(f"User {update.effective_user.id} created swarm for: {research_topic}")
            
        except Exception as e:
            await update.message.reply_text(f"❌ Failed to create swarm: {str(e)}")
            self.logger.error(f"Swarm creation failed: {e}")
    
    async def compete_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /compete command for agent competition"""
        if len(context.args) < 3:
            await update.message.reply_text(
                "❌ Usage: `/compete <agent1_id> <agent2_id> <task>`\n"
                "Example: `/compete researcher_123 analyst_456 \"analyze blockchain trends\"`",
                parse_mode='Markdown'
            )
            return
        
        agent1_id = context.args[0]
        agent2_id = context.args[1]
        task = " ".join(context.args[2:]).strip('"')
        
        try:
            # Verify agents exist
            agent1 = self.agi_system.agent_registry.get_agent(agent1_id)
            agent2 = self.agi_system.agent_registry.get_agent(agent2_id)
            
            if not agent1 or not agent2:
                await update.message.reply_text("❌ One or both agents not found. Use `/list` to see available agents.")
                return
            
            competition_msg = await update.message.reply_text(
                f"🏟️ **Agent Competition Started!**\n\n"
                f"🥊 **{agent1.name}** vs **{agent2.name}**\n"
                f"📋 Task: {task}\n\n"
                f"⏳ Agents are working..."
            )
            
            # Simulate competition (in real implementation, would run actual agents)
            await asyncio.sleep(2)
            
            # Generate responses
            response1 = await self.generate_agent_response(agent1, task)
            response2 = await self.generate_agent_response(agent2, task)
            
            # Simple scoring (in real implementation, would use sophisticated metrics)
            score1 = len(response1) + agent1.score * 10
            score2 = len(response2) + agent2.score * 10
            
            winner = agent1 if score1 > score2 else agent2
            loser = agent2 if score1 > score2 else agent1
            
            results_message = f"""
🏆 **Competition Results**

**Task:** {task}

🥇 **Winner: {winner.name}**
📊 Score: {max(score1, score2):.1f}
💭 Response: {(response1 if winner == agent1 else response2)[:200]}...

🥈 **Runner-up: {loser.name}**
📊 Score: {min(score1, score2):.1f}
💭 Response: {(response2 if winner == agent1 else response1)[:200]}...

🎯 **Victory Margin:** {abs(score1 - score2):.1f} points
            """
            
            await competition_msg.edit_text(results_message, parse_mode='Markdown')
            
            # Create vote keyboard for user feedback
            keyboard = [
                [
                    InlineKeyboardButton(f"👍 {agent1.name} Better", callback_data=f"vote_{agent1_id}"),
                    InlineKeyboardButton(f"👍 {agent2.name} Better", callback_data=f"vote_{agent2_id}")
                ],
                [
                    InlineKeyboardButton("🔄 Rematch", callback_data=f"rematch_{agent1_id}_{agent2_id}_{task}"),
                    InlineKeyboardButton("📊 Both Agents", callback_data="list_agents")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                "🗳️ Vote for the better response:",
                reply_markup=reply_markup
            )
            
            self.logger.info(f"Competition between {agent1_id} and {agent2_id} completed")
            
        except Exception as e:
            await update.message.reply_text(f"❌ Competition failed: {str(e)}")
            self.logger.error(f"Competition error: {e}")
    
    async def system_status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /system command for overall system status"""
        try:
            status = self.agi_system.get_system_status()
            
            # Calculate additional metrics
            uptime_emoji = "🟢" if status['avg_performance'] > 75 else "🟡" if status['avg_performance'] > 50 else "🔴"
            
            status_message = f"""
{uptime_emoji} **AGI System Status**

🤖 **Agents:** {status['active_agents']}/{status['total_agents']} active
📊 **Avg Performance:** {status['avg_performance']:.1f}%
✅ **Total Tasks:** {status['total_tasks_completed']}
🕐 **Status:** {status['system_uptime']}
🔄 **Last Update:** {status['last_updated'][:19]}

🏭 **System Health:**
{'🟢 Excellent' if status['avg_performance'] > 80 else '🟡 Good' if status['avg_performance'] > 60 else '🔴 Needs Attention'}

📈 **Capacity:**
Agent Slots: {10 - status['total_agents']}/10 available
Task Queue: Processing normally
Memory Usage: Optimal
            """
            
            # Create system action keyboard
            keyboard = [
                [
                    InlineKeyboardButton("🤖 View Agents", callback_data="list_agents"),
                    InlineKeyboardButton("🆕 Create Agent", callback_data="help_spawn")
                ],
                [
                    InlineKeyboardButton("🧪 System Test", callback_data="system_test"),
                    InlineKeyboardButton("🔄 Refresh", callback_data="system_status")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                status_message,
                parse_mode='Markdown',
                reply_markup=reply_markup
            )
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error getting system status: {str(e)}")
            self.logger.error(f"System status error: {e}")
    
    async def test_agent_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /test command for agent testing"""
        if not context.args:
            await update.message.reply_text(
                "❌ Usage: `/test <agent_id>`\n"
                "Use `/list` to see available agents.",
                parse_mode='Markdown'
            )
            return
        
        agent_id = context.args[0]
        
        try:
            agent = self.agi_system.agent_registry.get_agent(agent_id)
            if not agent:
                await update.message.reply_text(f"❌ Agent `{agent_id}` not found.")
                return
            
            test_msg = await update.message.reply_text(
                f"🧪 **Testing {agent.name}**\n\n"
                f"Running performance tests..."
            )
            
            # Run test queries
            test_queries = [
                f"Demonstrate your {agent.agent_type} capabilities",
                f"Analyze current trends in {', '.join(agent.specializations)}",
                f"Provide insights on {agent.purpose}"
            ]
            
            test_results = []
            total_time = 0
            
            for i, query in enumerate(test_queries):
                start_time = time.time()
                response = await self.generate_agent_response(agent, query)
                execution_time = time.time() - start_time
                total_time += execution_time
                
                quality_score = min(100, len(response) / 10)  # Simple quality metric
                test_results.append({
                    "query": query,
                    "response_time": execution_time,
                    "quality_score": quality_score,
                    "response": response[:100] + "..."
                })
            
            avg_time = total_time / len(test_queries)
            avg_quality = sum(r["quality_score"] for r in test_results) / len(test_results)
            
            results_message = f"""
🧪 **Test Results for {agent.name}**

⚡ **Performance:**
Avg Response Time: {avg_time:.2f}s
Avg Quality Score: {avg_quality:.1f}/100
Test Coverage: {len(test_queries)} scenarios

📊 **Test Details:**
"""
            
            for i, result in enumerate(test_results, 1):
                results_message += f"""
**Test {i}:** {result['query'][:30]}...
⏱️ Time: {result['response_time']:.2f}s
📊 Quality: {result['quality_score']:.1f}/100
"""
            
            overall_score = (avg_quality + (100 - min(100, avg_time * 20))) / 2
            grade = "🏆 Excellent" if overall_score > 80 else "👍 Good" if overall_score > 60 else "⚠️ Needs Improvement"
            
            results_message += f"\n🎯 **Overall Grade:** {grade} ({overall_score:.1f}/100)"
            
            await test_msg.edit_text(results_message, parse_mode='Markdown')
            
            # Create follow-up keyboard
            keyboard = [
                [
                    InlineKeyboardButton("💬 Talk to Agent", callback_data=f"talk_{agent_id}"),
                    InlineKeyboardButton("📊 Agent Status", callback_data=f"status_{agent_id}")
                ],
                [
                    InlineKeyboardButton("🔄 Retest", callback_data=f"test_{agent_id}"),
                    InlineKeyboardButton("🏟️ Find Competitor", callback_data=f"find_competitor_{agent_id}")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                "What would you like to do next?",
                reply_markup=reply_markup
            )
            
            self.logger.info(f"Agent {agent_id} tested with score {overall_score:.1f}")
            
        except Exception as e:
            await update.message.reply_text(f"❌ Testing failed: {str(e)}")
            self.logger.error(f"Agent testing error: {e}")
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard callbacks"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        user_id = query.from_user.id
        
        try:
            if data.startswith("talk_"):
                agent_id = data[5:]
                self.active_agents[user_id] = agent_id
                agent = self.agi_system.agent_registry.get_agent(agent_id)
                await query.edit_message_text(
                    f"💬 **Now talking to {agent.name}**\n\n"
                    f"Just type your messages normally. Use /stop_talk to end conversation.\n\n"
                    f"👋 **{agent.name}:** Hello! How can I help you today?",
                    parse_mode='Markdown'
                )
            
            elif data.startswith("status_"):
                agent_id = data[7:]
                # Trigger status command
                context.args = [agent_id]
                await self.status_command(update, context)
            
            elif data.startswith("test_"):
                agent_id = data[5:]
                context.args = [agent_id]
                await self.test_agent_command(update, context)
            
            elif data == "list_agents":
                await self.list_agents_command(update, context)
            
            elif data == "system_status":
                await self.system_status_command(update, context)
            
            elif data == "end_chat":
                if user_id in self.active_agents:
                    del self.active_agents[user_id]
                await query.edit_message_text("👋 Conversation ended. Use /list to see agents or /help for commands.")
            
            elif data.startswith("vote_"):
                agent_id = data[5:]
                agent = self.agi_system.agent_registry.get_agent(agent_id)
                await query.edit_message_text(f"👍 Thanks for voting for {agent.name}! Your feedback helps improve the agents.")
            
            else:
                await query.edit_message_text("🤖 Action processed. Use /help for available commands.")
                
        except Exception as e:
            await query.edit_message_text(f"❌ Error: {str(e)}")
            self.logger.error(f"Callback error: {e}")
    
    async def task_agent_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /task command to assign task to agent"""
        if len(context.args) < 2:
            await update.message.reply_text(
                "❌ Usage: `/task <agent_id> <task_description>`\n"
                "Example: `/task researcher_123 \"analyze AI trends in healthcare\"`",
                parse_mode='Markdown'
            )
            return
        
        agent_id = context.args[0]
        task_description = " ".join(context.args[1:]).strip('"')
        
        try:
            agent = self.agi_system.agent_registry.get_agent(agent_id)
            if not agent:
                await update.message.reply_text(f"❌ Agent `{agent_id}` not found.")
                return
            
            # Assign task
            task_id = await self.agi_system.assign_task_to_agent(agent_id, task_description)
            
            await update.message.reply_text(
                f"✅ **Task Assigned!**\n\n"
                f"🤖 **Agent:** {agent.name}\n"
                f"📋 **Task:** {task_description}\n"
                f"🆔 **Task ID:** `{task_id}`\n\n"
                f"The agent is now working on your task. Use `/status {agent_id}` to check progress.",
                parse_mode='Markdown'
            )
            
            self.logger.info(f"Task {task_id} assigned to agent {agent_id}")
            
        except Exception as e:
            await update.message.reply_text(f"❌ Failed to assign task: {str(e)}")
            self.logger.error(f"Task assignment error: {e}")
    
    def run(self):
        """Run the Telegram bot"""
        print("🚀 Starting Telegram AGI Bot...")
        print(f"Bot will be available at @{self.app.bot.username if hasattr(self.app.bot, 'username') else 'YourBot'}")
        print("Press Ctrl+C to stop")
        
        self.logger.info("Telegram AGI Bot started")
        self.app.run_polling()

def main():
    """Main function to run Telegram bot"""
    # Get bot token from environment or config
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    
    if not token:
        print("❌ TELEGRAM_BOT_TOKEN environment variable not set!")
        print("\nTo set up:")
        print("1. Create a bot with @BotFather on Telegram")
        print("2. Set environment variable: export TELEGRAM_BOT_TOKEN='your_token_here'")
        print("3. Run this script again")
        return
    
    try:
        # Create and run bot
        bot = TelegramAGIBot(token)
        bot.run()
    except KeyboardInterrupt:
        print("\n⏹️ Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Bot error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
