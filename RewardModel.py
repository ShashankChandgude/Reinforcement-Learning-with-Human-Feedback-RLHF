import torch.nn as nn

class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        reward = self.reward_head(outputs.hidden_states[-1][:, -1, :])  # Use last hidden state
        return reward
