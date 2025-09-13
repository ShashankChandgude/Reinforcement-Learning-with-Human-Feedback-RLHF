# 🚀 **RLHF Pipeline - Quick Start Guide**

> **Get up and running with our complete RLHF pipeline in under 10 minutes!**

---

## 🎯 **Overview**

This guide will walk you through:
1. **Environment Setup** (2 minutes)
2. **Running Your First RLHF Pipeline** (5 minutes)  
3. **Understanding the Results** (2 minutes)
4. **Next Steps** (1 minute)

**Total Time**: ~10 minutes

---

## 📋 **Prerequisites**

Before you start, make sure you have:
- ✅ **Python 3.8+** installed
- ✅ **8GB+ RAM** (16GB recommended)
- ✅ **GPU with CUDA** (optional, but recommended)
- ✅ **Internet connection** (for downloading models/datasets)

---

## 🛠️ **Step 1: Environment Setup** *(2 minutes)*

### **1.1 Clone the Repository**
```bash
git clone https://github.com/your-username/rlhf-pipeline.git
cd rlhf-pipeline
```

### **1.2 Create Virtual Environment**
```bash
# Create virtual environment
python -m venv .venv

# Activate it
# On Windows:
.\.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate
```

### **1.3 Install Dependencies**
```bash
pip install -r requirements.txt
```

**Expected output**: Installation of PyTorch, Transformers, and other dependencies (~2-3 minutes)

---

## 🏃 **Step 2: Run Your First RLHF Pipeline** *(5 minutes)*

### **2.1 Test the Complete Pipeline**
```bash
python test_full_pipeline.py
```

**What this does**:
- ✅ Tests all pipeline components
- ✅ Trains a small reward model (20 examples)
- ✅ Runs LoRA fine-tuning
- ✅ Executes PPO training
- ✅ Generates evaluation metrics

**Expected runtime**: ~5-8 minutes

**Expected output**:
```
[INFO] Starting full RLHF pipeline test...
[INFO] Reward Model Training: [PASS] (329.2s)
[INFO] LoRA Training: [PASS] (106.2s)
[INFO] PPO Training: [PASS] (295.9s)
[INFO] Evaluation: [PASS] (33.1s)
[INFO] ALL TESTS PASSED! Full pipeline is working!
```

### **2.2 Run Individual Components** *(Optional)*

If you want to run components separately:

```bash
# Train reward model only
python run_full_pipeline.py --phase reward

# Run LoRA fine-tuning only  
python run_full_pipeline.py --phase lora

# Run PPO training only
python run_full_pipeline.py --phase ppo

# Run evaluation only
python run_full_pipeline.py --phase eval
```

---

## 📊 **Step 3: Understanding the Results** *(2 minutes)*

### **3.1 View Evaluation Results**
```bash
python evaluate_rlhf_pipeline.py
```

**Key metrics to look for**:
- **BERTScore: ~0.80** (Good semantic understanding)
- **BLEU Score: ~0.28** (Moderate text quality)
- **Preference Accuracy: ~30%** (Learning human preferences)
- **Average Reward: ~0.41** (Positive reward signal)

### **3.2 Check Generated Files**
```bash
# View results
cat evaluation/rlhf_report.md

# Check model artifacts
ls models/reward_model_preference/

# View training logs
ls logs/
```

### **3.3 Sample Model Responses**

The evaluation will show sample responses like:
```
Prompt: "What are some cuss words in english?"
Response: "I'm a little confused about the word 'cuss'..."
```
*(Notice how the model avoids harmful content - this shows safety alignment!)*

---

## 🎯 **Step 4: Next Steps** *(1 minute)*

### **🔧 Customize Your Training**

Edit configuration files to customize training:

```bash
# Edit reward model config
nano configs/reward_preference.yaml

# Edit LoRA config  
nano configs/lora_7b_config.yaml

# Edit PPO config
nano configs/ppo_preference.yaml
```

### **📈 Scale Up Your Training**

Ready to train on more data? Update configs:

```yaml
# In reward_preference.yaml
dataset:
  subset_size: 1000  # Increase from 20 to 1000
  
training:
  epochs: 5  # Increase from 1 to 5
```

### **🚀 Try Different Models**

```yaml
# In any config file
model: microsoft/DialoGPT-medium  # Larger model
# or
model: EleutherAI/gpt-neo-1.3B   # Even larger!
```

---

## 🐛 **Troubleshooting**

### **Common Issues & Solutions**

#### **Issue**: `ModuleNotFoundError: No module named 'torch'`
**Solution**: Make sure virtual environment is activated and dependencies installed:
```bash
.\.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

#### **Issue**: `CUDA out of memory`
**Solution**: Reduce batch size in configs:
```yaml
training:
  batch_size: 1  # Reduce from 2 to 1
```

#### **Issue**: Training is very slow
**Solution**: 
- Use smaller dataset: `subset_size: 10`
- Use smaller model: `model: EleutherAI/gpt-neo-125M`
- Reduce epochs: `epochs: 1`

#### **Issue**: `'<=' not supported between instances of 'float' and 'str'`
**Solution**: This was a known bug that's been fixed. Make sure you're using the latest version.

---

## 📚 **Understanding the Pipeline**

### **What Just Happened?**

1. **Reward Model Training**: The model learned to predict human preferences
2. **LoRA Fine-tuning**: Memory-efficient parameter updates (only 0.23% of parameters!)
3. **PPO Training**: Policy optimization using the reward model
4. **Evaluation**: Comprehensive assessment of the aligned model

### **Key Files Created**:
- `models/reward_model_preference/` - Trained reward model
- `evaluation/rlhf_results.json` - Detailed metrics
- `evaluation/rlhf_report.md` - Human-readable report
- `logs/` - Training logs with timestamps

---

## 🎓 **Learning Resources**

### **Understanding RLHF**
- [OpenAI's RLHF Paper](https://arxiv.org/abs/2203.02155)
- [Anthropic's Constitutional AI](https://arxiv.org/abs/2212.08073)
- [Hugging Face RLHF Blog](https://huggingface.co/blog/rlhf)

### **Technical Deep Dives**
- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Parameter-efficient fine-tuning
- [PPO Paper](https://arxiv.org/abs/1707.06347) - Policy optimization
- [InstructGPT Paper](https://arxiv.org/abs/2203.02155) - RLHF methodology

---

## 💡 **Pro Tips**

### **🚀 Speed Up Training**
```bash
# Use test configs for faster iteration
python run_full_pipeline.py --config configs/test_reward_config.yaml
```

### **📊 Monitor Training**
```bash
# Watch logs in real-time
tail -f logs/preference_reward_*.log
```

### **🔬 Experiment Tracking**
```bash
# Enable Weights & Biases (optional)
wandb login
python run_full_pipeline.py --use-wandb
```

### **🎯 Focus on Specific Components**
```bash
# Just test reward model
python -c "from test_full_pipeline import test_reward_model_training; test_reward_model_training()"

# Just test LoRA
python -c "from test_full_pipeline import test_lora_training; test_lora_training()"
```

---

## 🎉 **Congratulations!**

You've successfully:
- ✅ **Set up** a complete RLHF pipeline
- ✅ **Trained** a reward model on human preferences  
- ✅ **Fine-tuned** a language model with LoRA
- ✅ **Optimized** the policy with PPO
- ✅ **Evaluated** the aligned model

**Your model is now more aligned with human preferences!**

---

## 🚀 **What's Next?**

### **Phase 5: Advanced Features**
- 🎮 **Interactive Demo**: Build a Gradio interface
- 🛡️ **Safety Evaluation**: Test for toxicity and bias
- 🔬 **Advanced Techniques**: Try DPO, Constitutional AI
- 🐳 **Deployment**: Docker containerization

### **Scale Up Your Training**
- 📈 **Larger Models**: 7B+ parameter models
- 📊 **More Data**: 1000+ preference examples
- ⚡ **Better Hardware**: Multi-GPU training
- 🔬 **Advanced Evaluation**: Human evaluation

---

## 📞 **Need Help?**

- **Issues**: [GitHub Issues](https://github.com/your-username/rlhf-pipeline/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/rlhf-pipeline/discussions)
- **Documentation**: Check the main [README.md](README.md)

---

<div align="center">

**🎯 Ready to build the next generation of aligned AI? Let's go!**

[⬅️ Back to Main README](README.md) | [➡️ Advanced Configuration](CONFIGURATION.md)

</div>
