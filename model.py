# model.py
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Deep Q-Network cho recommendation
        
        Args:
            state_dim: Số chiều của state vector
            action_dim: Số lượng actions (sản phẩm)
            hidden_dim: Số neurons trong hidden layers
        """
        super(DQN, self).__init__()
        
        # 3-layer fully connected network
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: State tensor (batch_size, state_dim)
            
        Returns:
            Q-values tensor (batch_size, action_dim)
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No activation on output (Q-values)
        return x
