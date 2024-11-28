import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from RewardModel import RewardModel
from dataloader import load_dolly_dataset
from torch.utils.data import DataLoader

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
EPOCHS = 1
MAX_LENGTH = 512
LEARNING_RATE = 1e-5
PPO_CLIP_EPSILON = 0.2

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M").to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token  # Ensure tokenizer has a pad token

# Define reward model
reward_model = RewardModel(model).to(DEVICE)

# Load Dolly dataset
print("Loading Dolly dataset...")
dataset = load_dolly_dataset(tokenizer, subset_size=5000, max_length=MAX_LENGTH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Optimizers
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
reward_optimizer = torch.optim.AdamW(reward_model.parameters(), lr=LEARNING_RATE)


# Supervised Fine-Tuning
def train_model(model, dataloader, optimizer, epochs):
    print("Starting Supervised Fine-Tuning...")
    model.train()
    for epoch in range(epochs):
        for step, (input_ids, attention_mask) in enumerate(dataloader):
            input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1} completed. Loss: {loss.item()}")


# Reward Model Training
def train_reward_model(reward_model, dataloader, optimizer, epochs):
    print("Training Reward Model...")
    reward_model.train()
    for epoch in range(epochs):
        for step, (input_ids, attention_mask) in enumerate(dataloader):
            input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
            rewards = reward_model(input_ids, attention_mask=attention_mask)
            loss = -torch.mean(rewards)  # Simple reward maximization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1} completed. Reward Model Loss: {loss.item()}")


# PPO Training
def ppo_training(model, reward_model, dataloader, optimizer, epochs, clip_epsilon=PPO_CLIP_EPSILON):
    print("Starting PPO Training...")
    model.train()
    for epoch in range(epochs):
        for step, (input_ids, attention_mask) in enumerate(dataloader):
            input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)

            # Generate predictions
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            actions = torch.argmax(probs, dim=-1)

            # Calculate rewards
            rewards = reward_model(input_ids, attention_mask=attention_mask)
            advantage = rewards - rewards.mean()

            # PPO loss calculation
            old_probs = torch.gather(probs, -1, actions.unsqueeze(-1)).squeeze(-1)
            new_probs = torch.gather(probs, -1, actions.unsqueeze(-1)).squeeze(-1)
            ratio = new_probs / (old_probs + 1e-10)
            clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
            ppo_loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()

            optimizer.zero_grad()
            ppo_loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1} completed. PPO Loss: {ppo_loss.item()}")


# Run Training
print("Step 1: Supervised Fine-Tuning")
train_model(model, dataloader, optimizer, EPOCHS)

print("Step 2: Reward Model Training")
train_reward_model(reward_model, dataloader, reward_optimizer, EPOCHS)

print("Step 3: Reinforcement Learning with PPO")
ppo_training(model, reward_model, dataloader, optimizer, EPOCHS)

# Save Models
model.save_pretrained("models/rlhf_gpt_neo_model")
tokenizer.save_pretrained("models/rlhf_gpt_neo_model")
torch.save(reward_model.state_dict(), "models/rlhf_reward_model.pth")

print("Training complete. Models saved successfully!")
