# agent.py
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from model import DQN
from replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=0.001, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_model = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = ReplayBuffer()
        self.gamma = gamma
        self.action_dim = action_dim
        
        # Epsilon strategy
        self.epsilon = 0.5  # Bắt đầu 50-50 sau khi có model
        self.epsilon_min = 0.3  # Dừng ở 30% exploration để cân bằng khám phá sản phẩm mới
        self.epsilon_decay = 0.998
        
        # Tracking
        self.is_trained = False  # Chưa có model
        self.train_count = 0
    
    def select_action(self, state):
        """Chọn 1 action (dùng cho training)"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.model.fc3.out_features)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.model(state_tensor)
        return q_values.argmax().item()
    
    def select_top_actions(self, state, k=10):
        """
        Chọn top k sản phẩm cho recommendation
        - Chưa có model (is_trained=False): Random 100%
        - Có model: Epsilon-greedy (epsilon% random, (1-epsilon)% model)
        """
        # COLD START: Chưa train lần nào → Random hoàn toàn
        if not self.is_trained:
            return list(np.random.choice(self.action_dim, size=k, replace=False))
        
        # ĐÃ CÓ MODEL: Epsilon-greedy strategy
        if np.random.rand() < self.epsilon:
            # EXPLORATION: Random k products
            return list(np.random.choice(self.action_dim, size=k, replace=False))
        else:
            # EXPLOITATION: Dùng model để chọn top k
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.model(state_tensor)
            
            # Lấy top k actions có Q-value cao nhất
            k = min(k, q_values.shape[1])
            top_k_indices = torch.topk(q_values, k, dim=1)[1]
            
            result = top_k_indices.squeeze().cpu().numpy()
            if result.ndim == 0:
                return [int(result.item())]
            return [int(x) for x in result.tolist()]
    
    def train_step(self, batch_size=32):
        """
        Training từ replay memory
        Sau train lần đầu tiên: is_trained = True → bắt đầu dùng model
        
        Chiến lược:
        - Nếu memory < batch_size: Train trên toàn bộ memory (online learning)
        - Nếu memory >= batch_size: Train trên batch ngẫu nhiên (experience replay)
        """
        if len(self.memory) < 1:
            return  # Không có data để train
        
        # Train ngay cả khi chỉ có 1 experience
        actual_batch_size = min(batch_size, len(self.memory))
        
        states, actions, rewards, next_states, dones = self.memory.sample(actual_batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        with torch.no_grad():
            q_next = self.target_model(next_states).max(1, keepdim=True)[0]
            q_target = rewards + self.gamma * q_next * (1 - dones)
        
        q_values = self.model(states).gather(1, actions)
        loss = nn.MSELoss()(q_values, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Đánh dấu đã train → có thể dùng model
        if not self.is_trained:
            self.is_trained = True
        
        self.train_count += 1
        
        # Giảm epsilon dần (50% → 10%)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def save_model(self, path="dqn_model.pt"):
        """Lưu model và optimizer state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load_model(self, path="dqn_model.pt"):
        """Load model và optimizer state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        # Đánh dấu đã có model
        self.is_trained = True
