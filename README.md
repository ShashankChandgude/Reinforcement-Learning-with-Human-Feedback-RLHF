# rlhf_ai_ml
Reinforcement Learning with Human Feedback (RLHF) for Educational AI
Overview
This project implements Reinforcement Learning with Human Feedback (RLHF) using GPT-Neo fine-tuned on the Dolly 15k dataset. The goal is to align AI-generated responses with human preferences in educational contexts. The approach integrates:

Supervised Fine-Tuning (SFT): Initial model training using prompt-response pairs.
Reward Model (RM): Trained to score responses based on human preference.
Proximal Policy Optimization (PPO): Reinforcement learning to optimize the language model using the reward signal.
Key Features
Human-Aligned Outputs: Improved factual accuracy and coherence of AI responses.
Generalization Across Domains: Robustness to out-of-distribution (OOD) prompts.
Efficient Training: Gradient accumulation and optimized batch sizes for GPU scalability.

Installation
1. Clone the Repository
bash
Copy code
git clone https://github.com/your-username/rlhf_ai_ml.git
cd rlhf_ai_ml
2. Install Dependencies
bash
Copy code
pip install -r requirements.txt
3. Download Dolly Dataset
The project uses the Dolly 15k dataset. It is automatically downloaded using Hugging Face Datasets.

File Structure
graphql
Copy code
rlhf_ai_ml/
├── main.py               # Main script for running SFT, RM, and PPO
├── reward_model.py       # Definition of the custom Reward Model
├── dataloader.py         # Custom DataLoader for tokenizing and loading Dolly dataset
├── models/               # Directory for saving trained models
│   ├── rlhf_gpt_neo_model/
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   ├── tokenizer.json
│   ├── rlhf_reward_model.pth
├── requirements.txt      # Python dependencies
└── README.md             # Documentation
How to Run
Supervised Fine-Tuning

bash
Copy code
python main.py --phase sft
Reward Model Training

bash
Copy code
python main.py --phase reward
Reinforcement Learning with PPO

bash
Copy code
python main.py --phase ppo
Test Model

bash
Copy code
python main.py --phase test
Results
Key Metrics
BLEU Score: Measures the similarity between model responses and human-preferred outputs.
Reward Alignment: Quantifies how well the outputs match human feedback.
Out-of-Distribution Robustness: Evaluates performance on unseen prompts.
Visualizations
Bar Graph: Reward Scores Across Baselines
Random Responses (RR), Supervised Fine-Tuning (SFT), Reward-Only (RO), RLHF (SPP).
Line Graph: Model Performance Across OOD Sensitivity
Compares robustness of RLHF vs. baselines.
Include images:

Pre-Trained Models
Pre-trained models are available for download:

Fine-Tuned GPT-Neo: Download Here
Reward Model: Download Here
Challenges and Insights
Challenges
High GPU memory usage during reward model training.
Ensuring factual accuracy in outputs using human preferences.
Managing long training times on limited resources.
Insights
RLHF effectively balances exploration and exploitation, improving coherence.
Combining SFT and PPO leads to significant performance improvements.
Future Work
Experimenting with larger models like GPT-3.
Incorporating advanced toxicity filtering mechanisms.
Expanding datasets to include diverse educational queries.
Contributing
We welcome contributions to this project! Follow these steps:

Fork the repository.
Create a new branch:
bash
Copy code
git checkout -b feature/new-feature
Commit your changes:
bash
Copy code
git commit -m "Add new feature"
Push to your fork:
bash
Copy code
git push origin feature/new-feature
Open a Pull Request.
License
This project is licensed under the MIT License.

Acknowledgements
Hugging Face Transformers for model and tokenizer integration.
Databricks Dolly 15k for providing the dataset.
PyTorch Framework for building and training the models.

