"""
Quick Demo: AI Image Prompt Generator
====================================
Standalone script to test the image prompt generation functionality
without launching the full Streamlit dashboard.
"""

import sys
import os
from pathlib import Path

# Add the models directory to the path
sys.path.append(str(Path(__file__).parent / "models"))

try:
    from system_prompt_injector import SystemPromptInjector
    SYSTEM_PROMPT_AVAILABLE = True
except ImportError:
    SYSTEM_PROMPT_AVAILABLE = False
    print("⚠️  SystemPromptInjector not available, using basic enhancement")

class SimplePromptGenerator:
    """Simplified version of the prompt generator for testing"""
    
    def __init__(self):
        self.system_prompt_injector = None
        if SYSTEM_PROMPT_AVAILABLE:
            try:
                self.system_prompt_injector = SystemPromptInjector()
                print("✅ SystemPromptInjector loaded successfully")
            except Exception as e:
                print(f"⚠️  SystemPromptInjector initialization failed: {e}")
    
    def generate_image_prompt(self, user_input: str, style: str = "Photorealistic", complexity: str = "Detailed") -> str:
        """Generate enhanced image prompts"""
        
        # Base prompt enhancement
        style_prompts = {
            "Photorealistic": "ultra-realistic, 8K resolution, professional photography, cinematic lighting",
            "Artistic": "artistic masterpiece, vibrant colors, creative composition, museum quality",
            "Fantasy": "magical fantasy artwork, ethereal lighting, mystical atmosphere, enchanted",
            "Sci-Fi": "futuristic sci-fi concept, advanced technology, neon lighting, cyberpunk aesthetic",
            "Abstract": "abstract art, geometric patterns, surreal composition, contemporary style"
        }
        
        complexity_modifiers = {
            "Simple": "clean, minimalist, simple composition",
            "Detailed": "highly detailed, intricate patterns, complex composition, rich textures",
            "Complex": "extremely detailed, multiple layers, complex scene, photorealistic details, professional"
        }
        
        # Get style and complexity modifiers
        style_mod = style_prompts.get(style, "high quality artwork")
        complexity_mod = complexity_modifiers.get(complexity, "detailed")
        
        # Enhanced prompt using system prompt injection techniques
        if self.system_prompt_injector:
            try:
                # Use the system prompt injector to enhance the prompt
                enhanced_input = self.system_prompt_injector.enhance_prompt(
                    user_input, 
                    task_type="creative"
                )
            except Exception as e:
                print(f"⚠️  SystemPromptInjector enhancement failed: {e}")
                enhanced_input = user_input
        else:
            enhanced_input = user_input
        
        # Construct the final prompt
        final_prompt = f"{enhanced_input}, {style_mod}, {complexity_mod}, award-winning, trending on artstation"
        
        # Add technical parameters
        technical_params = "perfect composition, rule of thirds, golden ratio, sharp focus, depth of field"
        
        return f"{final_prompt}, {technical_params}"

def main():
    """Demo the prompt generator"""
    
    print("🎨 AI Image Prompt Generator - Quick Demo")
    print("=" * 50)
    
    # Initialize generator
    generator = SimplePromptGenerator()
    
    # Demo prompts
    demo_prompts = [
        {
            "input": "A majestic dragon flying over a medieval castle at sunset",
            "style": "Fantasy",
            "complexity": "Complex"
        },
        {
            "input": "A robot working in a futuristic laboratory",
            "style": "Sci-Fi",
            "complexity": "Detailed"
        },
        {
            "input": "A peaceful mountain lake with reflection",
            "style": "Photorealistic",
            "complexity": "Detailed"
        },
        {
            "input": "Abstract geometric patterns in vibrant colors",
            "style": "Abstract",
            "complexity": "Simple"
        }
    ]
    
    # Generate and display enhanced prompts
    for i, demo in enumerate(demo_prompts, 1):
        print(f"\n🎯 Demo {i}: {demo['style']} Style, {demo['complexity']} Detail")
        print(f"📝 Input: {demo['input']}")
        print("🔄 Generating enhanced prompt...")
        
        enhanced = generator.generate_image_prompt(
            demo['input'], 
            demo['style'], 
            demo['complexity']
        )
        
        print(f"✨ Enhanced Prompt:")
        print(f"   {enhanced}")
        print("-" * 80)
    
    # Interactive mode
    print("\n🎮 Interactive Mode - Enter your own prompts!")
    print("(Type 'quit' to exit)\n")
    
    while True:
        try:
            user_input = input("📝 Enter your image concept: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                print("⚠️  Please enter a description.")
                continue
            
            # Get style preference
            print("\n🎨 Available styles:")
            styles = ["Photorealistic", "Artistic", "Fantasy", "Sci-Fi", "Abstract"]
            for i, style in enumerate(styles, 1):
                print(f"   {i}. {style}")
            
            style_choice = input("Select style (1-5, default=1): ").strip()
            try:
                style_idx = int(style_choice) - 1 if style_choice else 0
                style = styles[style_idx] if 0 <= style_idx < len(styles) else styles[0]
            except (ValueError, IndexError):
                style = "Photorealistic"
            
            # Get complexity preference
            print("\n🔧 Detail levels:")
            complexities = ["Simple", "Detailed", "Complex"]
            for i, comp in enumerate(complexities, 1):
                print(f"   {i}. {comp}")
            
            comp_choice = input("Select detail level (1-3, default=2): ").strip()
            try:
                comp_idx = int(comp_choice) - 1 if comp_choice else 1
                complexity = complexities[comp_idx] if 0 <= comp_idx < len(complexities) else complexities[1]
            except (ValueError, IndexError):
                complexity = "Detailed"
            
            print(f"\n🔄 Generating {style} prompt with {complexity} details...")
            
            enhanced = generator.generate_image_prompt(user_input, style, complexity)
            
            print(f"\n✨ Enhanced Prompt:")
            print(f"{enhanced}")
            
            # Save option
            save = input("\n💾 Save this prompt to file? (y/n): ").strip().lower()
            if save in ['y', 'yes']:
                filename = f"enhanced_prompt_{int(time.time())}.txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"Original: {user_input}\n")
                    f.write(f"Style: {style}\n")
                    f.write(f"Complexity: {complexity}\n")
                    f.write(f"Enhanced: {enhanced}\n")
                print(f"✅ Saved to {filename}")
            
            print("\n" + "="*80)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Thanks for using the AI Image Prompt Generator!")

if __name__ == "__main__":
    import time
    main()
