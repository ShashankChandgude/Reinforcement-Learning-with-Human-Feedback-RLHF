#!/usr/bin/env python3
"""
Quick launcher for RLHF Demo
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed."""
    try:
        import gradio
        import torch
        import transformers
        print("All required packages are installed")
        return True
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install requirements: pip install -r requirements_demo.txt")
        return False

def main():
    """Main launcher function."""
    print("RLHF Demo Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("demo_app.py"):
        print("demo_app.py not found. Please run from project root directory.")
        return
    
    # Check requirements
    if not check_requirements():
        return
    
    # Check if models exist
    model_paths = [
        "models/ppo_preference_balanced",
        "models/supervised_rlhf_model",
        "models/reward_model_preference_balanced"
    ]
    
    missing_models = []
    for path in model_paths:
        if not os.path.exists(path):
            missing_models.append(path)
    
    if missing_models:
        print("Warning: Some trained models are missing:")
        for path in missing_models:
            print(f"   - {path}")
        print("\nThe demo will work with base model only.")
        print("To train models, run: python run_full_pipeline.py --phase full")
        print()
    
    # Launch demo
    print("Starting RLHF Demo...")
    print("The demo will open in your browser automatically at http://localhost:7860")
    print("Windows Security Note:")
    print("   - If Windows Defender shows 'Potentially Unwanted App', click 'Allow'")
    print("   - This is normal for local web servers - Gradio is safe")
    print("Press Ctrl+C to stop the demo when you're done testing")
    print()
    
    try:
        subprocess.run([sys.executable, "demo_app.py"], check=True)
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    except Exception as e:
        print(f"Error running demo: {e}")

if __name__ == "__main__":
    main()
