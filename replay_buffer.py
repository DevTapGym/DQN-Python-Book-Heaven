# replay_buffer.py
import random
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample batch với ưu tiên cho experience mới nhất
        - Luôn lấy experience mới nhất (most recent)
        - Random (batch_size - 1) experiences cũ
        """
        if len(self.memory) == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        if len(self.memory) == 1:
            # Chỉ có 1 experience → trả về nó
            batch = [self.memory[-1]]
        elif len(self.memory) <= batch_size:
            # Ít hơn batch_size → lấy tất cả
            batch = list(self.memory)
        else:
            # Đủ data: Lấy newest + random (batch_size-1) cũ
            newest = self.memory[-1]  # Experience mới nhất
            
            # Random (batch_size - 1) experiences từ phần còn lại
            # (không bao gồm experience mới nhất)
            old_experiences = list(self.memory)[:-1]  # Tất cả trừ cái cuối
            sampled_old = random.sample(old_experiences, batch_size - 1)
            
            # Kết hợp: newest ở đầu + các old experiences
            batch = [newest] + sampled_old
        
        # Unpack batch
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done))
    
    def __len__(self):
        return len(self.memory)
