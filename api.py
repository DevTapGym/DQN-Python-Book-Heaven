from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import numpy as np
from agent import DQNAgent
from state_encoder import encode_state
from config import MAX_PRODUCTS, CATEGORIES, GENDER_MAP, AGE_MAP, POSITION_MAP

app = FastAPI()

# STATE_DIM calculation:
# Common: 3(gender) + 5(age) + 7(day) = 15
# Position-specific (max): 1 + 5 + 50 + 10 + 3 = 69 (search/cart/home đều 69)
# Position encoding: 3 (one-hot)
# Total: 15 + 69 + 3 = 87
STATE_DIM = 87
ACTION_DIM = MAX_PRODUCTS

agent = DQNAgent(STATE_DIM, ACTION_DIM)

# Data model cho /recommend
class RecommendInput(BaseModel):
    raw_data: dict
    position: str
    
    @validator('position')
    def validate_position(cls, v):
        if v not in POSITION_MAP:
            raise ValueError(f"position phải là một trong: {list(POSITION_MAP.keys())}")
        return v
    
    @validator('raw_data')
    def validate_raw_data(cls, v):
        # Gender validation
        if 'gender' in v and v['gender'] not in GENDER_MAP:
            raise ValueError(f"gender phải là một trong: {list(GENDER_MAP.keys())}")
        
        # Age group validation
        if 'age_group' in v and v['age_group'] not in AGE_MAP:
            raise ValueError(f"age_group phải là một trong: {list(AGE_MAP.keys())}")
        
        # Day of week validation
        if 'day_of_week' in v:
            day = v['day_of_week']
            if not isinstance(day, int) or day < 1 or day > 7:
                raise ValueError("day_of_week phải là số nguyên từ 1-7")
        
        # Category validation
        if 'category' in v:
            cats = v['category'] if isinstance(v['category'], list) else [v['category']]
            invalid_cats = [c for c in cats if c not in CATEGORIES]
            if invalid_cats:
                raise ValueError(f"category không hợp lệ: {invalid_cats}. Phải nằm trong: {CATEGORIES}")
        
        # Product IDs validation
        if 'product_ids' in v:
            products = v['product_ids'] if isinstance(v['product_ids'], list) else [v['product_ids']]
            for pid in products:
                if not isinstance(pid, int) or pid < 1 or pid > MAX_PRODUCTS:
                    raise ValueError(f"product_id phải là số nguyên từ 1-{MAX_PRODUCTS}")
        
        # Numeric fields validation
        if 'num_products' in v:
            if not isinstance(v['num_products'], (int, float)) or v['num_products'] < 0:
                raise ValueError("num_products phải là số không âm")
        
        if 'total_value' in v:
            if not isinstance(v['total_value'], (int, float)) or v['total_value'] < 0:
                raise ValueError("total_value phải là số không âm")
        
        if 'avg_value' in v:
            if not isinstance(v['avg_value'], (int, float)) or v['avg_value'] < 0:
                raise ValueError("avg_value phải là số không âm")
        
        if 'recent_searches' in v:
            if not isinstance(v['recent_searches'], (int, float)) or v['recent_searches'] < 0:
                raise ValueError("recent_searches phải là số không âm")
        
        return v

@app.post("/recommend")
def recommend(input: RecommendInput):
    """
    Gợi ý top 10 sản phẩm dựa trên state
    
    Strategy:
    - Chưa có model: Random 100%
    - Có model: Epsilon-greedy (epsilon% random, (1-epsilon)% model)
    - Epsilon giảm dần: 50% → 10% theo thời gian
    """
    try:
        # Encode state từ dữ liệu thô
        state = encode_state(input.raw_data, input.position)
        top_actions = agent.select_top_actions(state, k=10)
        
        # Convert từ index (0-49) sang product ID (1-50)
        # Ensure conversion to Python int (not numpy.int32)
        product_ids = [int(action) + 1 for action in top_actions]
        
        return {
            "recommended_products": product_ids,
            "count": len(product_ids),
            "strategy": "random" if not agent.is_trained else f"epsilon-greedy (ε={agent.epsilon:.2f})",
            "model_status": "trained" if agent.is_trained else "cold_start"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi xử lý: {str(e)}")

# Data model cho /train
class TrainInput(BaseModel):
    raw_data: dict
    position: str
    action: int = Field(..., ge=1, le=MAX_PRODUCTS, description=f"Product ID phải từ 1-{MAX_PRODUCTS}")
    reward: float = Field(..., description="Reward value (có thể âm hoặc dương)")
    next_raw_data: dict
    next_position: str
    done: bool
    
    @validator('position', 'next_position')
    def validate_positions(cls, v):
        if v not in POSITION_MAP:
            raise ValueError(f"position phải là một trong: {list(POSITION_MAP.keys())}")
        return v
    
    @validator('raw_data', 'next_raw_data')
    def validate_raw_data_train(cls, v):
        # Tái sử dụng validation logic từ RecommendInput
        return RecommendInput.validate_raw_data(v)

@app.post("/train")
def train_step(input: TrainInput):
    """
    Training model từ user feedback
    - Train ngay sau mỗi feedback
    - Auto update target network mỗi 100 lần train
    - Auto save model mỗi 500 lần train
    """
    try:
        # Encode state và next_state
        state = encode_state(input.raw_data, input.position)
        next_state = encode_state(input.next_raw_data, input.next_position)
        
        # Convert product ID (1-50) về action index (0-49)
        action_index = input.action - 1
        
        # Lưu vào memory
        agent.memory.push(state, action_index, input.reward, next_state, input.done)
        
        # Training
        agent.train_step()
        
        # Auto update target network mỗi 100 trains
        if agent.train_count % 100 == 0:
            agent.update_target()
        
        # Auto save model mỗi 500 trains
        if agent.train_count % 500 == 0:
            agent.save_model(f"dqn_checkpoint_{agent.train_count}.pt")
        
        return {
            "status": "trained",
            "epsilon": float(agent.epsilon),
            "memory_size": len(agent.memory),
            "train_count": int(agent.train_count),
            "model_activated": bool(agent.is_trained)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi training: {str(e)}")

@app.get("/status")
def get_status():
    """Lấy thông tin trạng thái agent"""
    return {
        "epsilon": float(agent.epsilon),
        "memory_size": len(agent.memory),
        "train_count": int(agent.train_count),
        "model_activated": bool(agent.is_trained),
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "strategy": "random (cold_start)" if not agent.is_trained else f"epsilon-greedy (ε={agent.epsilon:.2f})"
    }
