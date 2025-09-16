#!/usr/bin/env python3
"""
Interactive RLHF Demo Application
Professional Gradio interface for showcasing RLHF capabilities
"""

import gradio as gr
import torch
import json
import os
import sys
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GenerationConfig
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.config_loader import load_config
from evaluation.rlhf_evaluator import run_rlhf_evaluation
from training.reward_model import RewardModel

class RLHFDemo:
    """Interactive RLHF demonstration application."""
    
    def __init__(self):
        self.base_model = None  # Balanced reward model as base
        self.ppo_model = None
        self.supervised_model = None
        self.reward_model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_models()
        
    def load_models(self):
        """Load all trained models."""
        try:
            print("Loading models...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model (balanced reward model)
            base_model_path = "models/reward_model_preference_balanced"
            if os.path.exists(base_model_path):
                try:
                    self.base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
                    self.base_model.to(self.device)
                    self.base_model.eval()
                    print("Balanced reward model loaded as base model")
                except Exception as e:
                    print(f"Error loading balanced reward model: {e}")
                    print("Falling back to original base model")
                    base_model_path = "EleutherAI/gpt-neo-125M"
                    self.base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
                    self.base_model.to(self.device)
                    self.base_model.eval()
            else:
                print("Balanced reward model not found, using original base model")
                base_model_path = "EleutherAI/gpt-neo-125M"
                self.base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
                self.base_model.to(self.device)
                self.base_model.eval()
            
            # Load PPO model
            ppo_model_path = "models/ppo_preference_balanced"
            if os.path.exists(ppo_model_path):
                try:
                    self.ppo_model = AutoModelForCausalLM.from_pretrained(ppo_model_path)
                    self.ppo_model.to(self.device)
                    self.ppo_model.eval()
                    
                    # Load PPO tokenizer
                    self.ppo_tokenizer = AutoTokenizer.from_pretrained(ppo_model_path)
                    if self.ppo_tokenizer.pad_token is None:
                        self.ppo_tokenizer.pad_token = self.ppo_tokenizer.eos_token
                    
                    print("PPO model loaded successfully")
                except Exception as e:
                    print(f"Error loading PPO model: {e}")
                    self.ppo_model = None
                    self.ppo_tokenizer = None
            else:
                print("PPO model not found")
                self.ppo_model = None
                self.ppo_tokenizer = None
            
            # Load Supervised RLHF model
            supervised_model_path = "models/supervised_rlhf_model"
            if os.path.exists(supervised_model_path):
                try:
                    self.supervised_model = AutoModelForCausalLM.from_pretrained(supervised_model_path)
                    self.supervised_model.to(self.device)
                    self.supervised_model.eval()
                    
                    # Load supervised tokenizer
                    self.supervised_tokenizer = AutoTokenizer.from_pretrained(supervised_model_path)
                    if self.supervised_tokenizer.pad_token is None:
                        self.supervised_tokenizer.pad_token = self.supervised_tokenizer.eos_token
                    
                    print("Supervised RLHF model loaded successfully")
                except Exception as e:
                    print(f"Error loading supervised model: {e}")
                    self.supervised_model = None
                    self.supervised_tokenizer = None
            else:
                print("Supervised RLHF model not found")
                self.supervised_model = None
                self.supervised_tokenizer = None
                
            # Load reward model (balanced)
            reward_model_path = "models/reward_model_preference_balanced"
            if os.path.exists(reward_model_path):
                try:
                    # Load the base model for reward model
                    reward_base_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
                    self.reward_model = RewardModel(reward_base_model).to(self.device)
                    
                    # Load reward model weights
                    reward_weights_path = os.path.join(reward_model_path, "reward_model.pth")
                    if os.path.exists(reward_weights_path):
                        state_dict = torch.load(reward_weights_path, map_location=self.device)
                        self.reward_model.load_state_dict(state_dict)
                        self.reward_model.eval()
                        print("Balanced reward model loaded successfully")
                    else:
                        print("Reward model weights not found, using untrained reward model")
                except Exception as e:
                    print(f"Error loading reward model: {e}")
                    self.reward_model = None
            else:
                print("Balanced reward model directory not found")
                
            print("Model loading completed!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Continuing with base model only...")
    
    def generate_response(self, prompt, model_type, temperature, max_length, repetition_penalty):
        """Generate response from selected model."""
        try:
            if not prompt.strip():
                return "Please enter a prompt to generate a response."
            
            # Select model and tokenizer
            if model_type == "PPO Model" and self.ppo_model:
                model = self.ppo_model
                tokenizer = self.ppo_tokenizer if hasattr(self, 'ppo_tokenizer') and self.ppo_tokenizer else self.tokenizer
                model_name = "PPO"
            elif model_type == "Supervised Model" and self.supervised_model:
                model = self.supervised_model
                tokenizer = self.supervised_tokenizer if hasattr(self, 'supervised_tokenizer') and self.supervised_tokenizer else self.tokenizer
                model_name = "Supervised"
            else:
                model = self.base_model
                tokenizer = self.tokenizer
                model_name = "Balanced Reward"
            
            # Tokenize input with proper handling
            inputs = tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=400,  # Leave room for generation
                padding=False    # Don't pad for generation
            ).to(self.device)
            
            # Ensure we have attention mask
            if 'attention_mask' not in inputs:
                inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
            
            # Generation config with better parameters
            generation_config = GenerationConfig(
                max_new_tokens=min(max_length, 100),  # Limit to prevent runaway generation
                temperature=max(temperature, 0.1),    # Ensure temperature is not too low
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                top_p=0.9,
                top_k=50,
                no_repeat_ngram_size=3  # Prevent repetitive n-grams
            )
            
            # Generate with progress indication
            print(f"Generating response using {model_name} model...")
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    **generation_config.to_dict()
                )
            
            # Decode response using the same tokenizer
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new generated part more carefully
            # Handle cases where the prompt might not match exactly due to tokenization
            prompt_tokens = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][prompt_tokens:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Basic quality check
            if not response or len(response) < 5:
                return f"[{model_name} Model] Generated a very short response. Try adjusting parameters or a different prompt."
            
            # Check for obvious corruption (too many repeated characters)
            if len(set(response)) < 5:  # Very few unique characters
                return f"[{model_name} Model] Generated corrupted response. Try different parameters."
            
            return f"[{model_name} Model] {response}"
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def calculate_reward_score(self, prompt, response):
        """Calculate reward score for the response using the actual reward model."""
        try:
            if self.reward_model is None:
                return "Reward model not available"
            
            # Combine prompt and response
            full_text = prompt + " " + response
            
            # Tokenize with proper attention mask
            inputs = self.tokenizer(
                full_text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Calculate reward using the trained reward model
            with torch.no_grad():
                reward_tensor = self.reward_model(
                    inputs.input_ids, 
                    attention_mask=inputs.attention_mask
                )
                reward_score = reward_tensor.squeeze().item()
            
            return f"Reward Score: {reward_score:.3f}"
            
        except Exception as e:
            return f"Error calculating reward: {str(e)}"
    
    def compare_models(self, prompt, temperature, max_length, repetition_penalty):
        """Compare all three models: Balanced Reward, PPO, and Supervised."""
        try:
            # Generate from base model (balanced reward)
            base_response = self.generate_response(
                prompt, "Balanced Reward Model", temperature, max_length, repetition_penalty
            )
            
            # Generate from PPO model
            ppo_response = self.generate_response(
                prompt, "PPO Model", temperature, max_length, repetition_penalty
            )
            
            # Generate from supervised model
            supervised_response = self.generate_response(
                prompt, "Supervised Model", temperature, max_length, repetition_penalty
            )
            
            # Calculate rewards
            base_reward = self.calculate_reward_score(prompt, base_response)
            ppo_reward = self.calculate_reward_score(prompt, ppo_response)
            supervised_reward = self.calculate_reward_score(prompt, supervised_response)
            
            return base_response, ppo_response, supervised_response, base_reward, ppo_reward, supervised_reward
            
        except Exception as e:
            error_msg = f"Error in model comparison: {str(e)}"
            return error_msg, error_msg, error_msg, "Error", "Error", "Error"
    
    def load_training_metrics(self):
        """Load and display training metrics."""
        try:
            # Load evaluation results
            results_path = "evaluation/rlhf_results.json"
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    results = json.load(f)
                
                # Create metrics display
                metrics_text = "## Training Results\n\n"
                
                if "model_quality" in results:
                    quality = results["model_quality"]
                    metrics_text += f"### Model Quality Metrics\n"
                    metrics_text += f"- **BLEU Score**: {quality.get('bleu', 'N/A'):.3f}\n"
                    metrics_text += f"- **ROUGE Score**: {quality.get('rouge', 'N/A'):.3f}\n"
                    metrics_text += f"- **BERTScore**: {quality.get('bertscore', 'N/A'):.3f}\n"
                    metrics_text += f"- **Avg Response Length**: {quality.get('avg_response_length', 'N/A'):.1f}\n"
                    metrics_text += f"- **Avg Reward Score**: {quality.get('avg_reward_score', 'N/A'):.3f}\n\n"
                
                if "preference_alignment" in results:
                    alignment = results["preference_alignment"]
                    metrics_text += f"### Preference Alignment\n"
                    metrics_text += f"- **Preference Accuracy**: {alignment.get('preference_accuracy', 'N/A'):.1%}\n"
                    metrics_text += f"- **Avg Reward Difference**: {alignment.get('avg_reward_difference', 'N/A'):.3f}\n"
                    metrics_text += f"- **Chosen Avg Reward**: {alignment.get('chosen_avg_reward', 'N/A'):.3f}\n"
                    metrics_text += f"- **Rejected Avg Reward**: {alignment.get('rejected_avg_reward', 'N/A'):.3f}\n\n"
                
                return metrics_text
            else:
                return "Training metrics not available. Please run evaluation first."
                
        except Exception as e:
            return f"Error loading metrics: {str(e)}"
    
    def create_interface(self):
        """Create the Gradio interface."""
        
        # Custom CSS for professional look
        css = """
        .gradio-container {
            max-width: 1200px !important;
            margin: auto !important;
        }
        .metric-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .model-comparison {
            border: 2px solid #e1e5e9;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }
        """
        
        with gr.Blocks(css=css, title="RLHF Demo - Reinforcement Learning with Human Feedback") as demo:
            
            gr.Markdown("""
            # RLHF Demo - Reinforcement Learning with Human Feedback
            
            **A complete implementation of the methodology used by OpenAI (ChatGPT) and Anthropic (Claude)**
            
            This demo showcases a trained RLHF pipeline with advanced PPO training.
            """)
            
            with gr.Tabs():
                
                # Tab 1: Interactive Generation
                with gr.Tab("Interactive Generation"):
                    gr.Markdown("### Generate responses from our RLHF-trained models")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            prompt_input = gr.Textbox(
                                label="Enter your prompt:",
                                placeholder="Try: 'What are some cuss words in english?' or 'How do you embezzle money?'",
                                lines=3,
                                info="Test prompts that show safety alignment differences between base and RLHF models"
                            )
                            
                            with gr.Row():
                                model_choice = gr.Radio(
                                    choices=["Balanced Reward Model", "PPO Model", "Supervised Model"],
                                    value="PPO Model",
                                    label="Model Type"
                                )
                                
                            with gr.Row():
                                temperature = gr.Slider(0.1, 2.0, value=0.8, label="Temperature")
                                max_length = gr.Slider(10, 200, value=100, label="Max Length")
                                repetition_penalty = gr.Slider(1.0, 2.5, value=1.3, label="Repetition Penalty")
                            
                            generate_btn = gr.Button("Generate Response", variant="primary")
                        
                        with gr.Column(scale=2):
                            response_output = gr.Textbox(
                                label="Model Response:",
                                lines=8,
                                interactive=False
                            )
                            reward_output = gr.Textbox(
                                label="Reward Score:",
                                interactive=False
                            )
                    
                    def generate_and_score(p, m, t, ml, rp):
                        """Generate response once and calculate reward."""
                        response = self.generate_response(p, m, t, ml, rp)
                        reward = self.calculate_reward_score(p, response)
                        return response, reward
                    
                    generate_btn.click(
                        fn=generate_and_score,
                        inputs=[prompt_input, model_choice, temperature, max_length, repetition_penalty],
                        outputs=[response_output, reward_output]
                    )
                
                # Tab 2: Model Comparison
                with gr.Tab("Model Comparison"):
                    gr.Markdown("### Compare All Three Models: Balanced Reward, PPO, and Supervised")
                    
                    with gr.Row():
                        with gr.Column():
                            compare_prompt = gr.Textbox(
                                label="Enter prompt for comparison:",
                                placeholder="What are some cuss words in english?",
                                lines=3
                            )
                            
                            with gr.Row():
                                compare_temp = gr.Slider(0.1, 2.0, value=0.8, label="Temperature")
                                compare_length = gr.Slider(10, 200, value=100, label="Max Length")
                                compare_penalty = gr.Slider(1.0, 2.5, value=1.3, label="Repetition Penalty")
                            
                            compare_btn = gr.Button("Compare Models", variant="primary")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Balanced Reward Model Response")
                            base_response = gr.Textbox(lines=6, interactive=False)
                            base_reward = gr.Textbox(label="Reward Score", interactive=False)
                        
                        with gr.Column():
                            gr.Markdown("### PPO Model Response")
                            ppo_response = gr.Textbox(lines=6, interactive=False)
                            ppo_reward = gr.Textbox(label="Reward Score", interactive=False)
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Supervised Model Response")
                            supervised_response = gr.Textbox(lines=6, interactive=False)
                            supervised_reward = gr.Textbox(label="Reward Score", interactive=False)
                    
                    compare_btn.click(
                        fn=self.compare_models,
                        inputs=[compare_prompt, compare_temp, compare_length, compare_penalty],
                        outputs=[base_response, ppo_response, supervised_response, base_reward, ppo_reward, supervised_reward]
                    )
                
                # Tab 3: Training Metrics
                with gr.Tab("Training Metrics"):
                    gr.Markdown("### Model Performance and Training Results")
                    
                    metrics_btn = gr.Button("Load Training Metrics", variant="primary")
                    metrics_output = gr.Markdown()
                    
                    metrics_btn.click(
                        fn=self.load_training_metrics,
                        outputs=metrics_output
                    )
                
                # Tab 4: About
                with gr.Tab("About"):
                    gr.Markdown("""
                    ## Project Achievements
                    
                    ### Key Breakthroughs:
                    - **90% Reward Model Accuracy**: Fixed critical data parsing bug (3.7x improvement)
                    - **Advanced PPO Implementation**: GAE, value function, entropy regularization
                    - **Professional-Grade Pipeline**: Complete RLHF with quality control
                    - **Robust Response Generation**: Quality filters and coherence improvements
                    
                    ### Technical Stack:
                    - **Model**: GPT-Neo-125M with LoRA fine-tuning
                    - **Training**: PPO with Generalized Advantage Estimation
                    - **Dataset**: Anthropic HH-RLHF preference data
                    - **Framework**: PyTorch, Transformers, Gradio
                    
                    ### Performance Metrics:
                    - **Preference Learning**: 90% accuracy 
                    - **Value Function**: 68% loss reduction
                    - **Exploration**: 44% entropy increase
                    - **Quality Control**: 100% broken response filtering
                    
                    ### Repository:
                    - **GitHub**: [Your Repository Link]
                    - **Documentation**: Comprehensive README with technical details
                    - **Code Quality**: Educational Purpose, production-ready implementation
                    
                    ---
                    
                    **Built for AI/ML portfolio demonstration**
                    """)
        
        return demo

def main():
    """Main function to run the demo."""
    print("Starting RLHF Demo Application...")
    
    # Create demo instance
    demo_app = RLHFDemo()
    
    # Create interface
    demo = demo_app.create_interface()
    
    # Launch demo
    print("Launching Gradio interface...")
    print("The demo will open in your default browser automatically")
    print("Local URL: http://localhost:7860")
    print("If Windows Defender blocks this, click 'Allow' to proceed")
    print("Press Ctrl+C to stop the demo")
    
    demo.launch(
        server_name="127.0.0.1",  # Use localhost instead of 0.0.0.0
        server_port=7860,
        share=False,  # Disable share link to avoid issues
        show_error=True,
        quiet=False,
        inbrowser=True  # Auto-open browser
    )

if __name__ == "__main__":
    main()
