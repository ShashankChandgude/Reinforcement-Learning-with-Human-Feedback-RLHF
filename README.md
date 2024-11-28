# **Reinforcement Learning with Human Feedback (RLHF) for Educational AI**

---

## **Project Overview**
This project implements **Reinforcement Learning with Human Feedback (RLHF)** to align AI-generated responses with human preferences in educational settings. By combining **Supervised Fine-Tuning (SFT)**, a **Reward Model (RM)**, and **Proximal Policy Optimization (PPO)**, the project enhances response coherence, factual accuracy, and user alignment.

---

## **Features**
- **Human-Aligned Outputs:** Generates factually accurate and contextually relevant responses to educational prompts.
- **Generalization Across Domains:** Performs well on out-of-distribution (OOD) prompts like science, history, and biology.
- **Efficient Training Pipeline:** Optimized with gradient accumulation and smaller batch sizes for GPU efficiency.

---

## **Installation**

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/rlhf_ai_ml.git
cd rlhf_ai_ml
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Dataset**
This project uses the **Dolly 15k dataset**, which is automatically downloaded using Hugging Face Datasets.

---

## **How to Run**

### **1. Supervised Fine-Tuning**
Train the model with supervised fine-tuning on the Dolly 15k dataset:
```bash
python main.py --phase sft
```

### **2. Reward Model Training**
Train the reward model to score human-preferred outputs:
```bash
python main.py --phase reward
```

### **3. Reinforcement Learning with PPO**
Optimize the language model using Proximal Policy Optimization (PPO):
```bash
python main.py --phase ppo
```

### **4. Test the Model**
Evaluate the fine-tuned model on test prompts:
```bash
python main.py --phase test
```

---

## **Results**

### **Key Metrics**
- **BLEU Score:** Measures similarity between model outputs and human-preferred responses.
- **Reward Alignment:** Quantifies alignment with human feedback.
- **Out-of-Distribution Robustness:** Evaluates model performance on unseen prompts.

### **Visualizations**
- **Bar Graph:** Reward scores for baselines (Random Responses, SFT, PPO).
- **Line Graph:** Model performance on out-of-distribution prompts.

---

## **Pre-Trained Models**
Download pre-trained models for testing:
- **Fine-Tuned GPT-Neo Model:** [Download Here](https://huggingface.co/your-model-link)
- **Reward Model:** [Download Here](https://huggingface.co/your-reward-model-link)

---

## **Challenges and Insights**

### **Challenges**
- **High Memory Usage:** Required optimizations like gradient accumulation for GPU efficiency.
- **Toxicity Filtering:** Manual intervention needed for improved safety.
- **Long Training Time:** RLHF required several hours per epoch on a T4 GPU.

### **Insights**
- RLHF effectively balances exploration and exploitation, resulting in coherent responses.
- Combining SFT and PPO significantly improves reward alignment and factual accuracy.

---

## **Future Work**
- Experimenting with larger models like GPT-3.
- Improving toxicity filtering with advanced evaluation techniques.
- Expanding datasets to include a broader range of queries.

---

## **License**
This project is licensed under the [MIT License](LICENSE).

---

## **Acknowledgements**
- **Hugging Face Transformers** for model and tokenizer integration.
- **Databricks Dolly 15k Dataset** for fine-tuning.
- **PyTorch Framework** for model training and customization.
