# Kiến trúc Hệ thống DQN Recommendation

## 📐 Tổng quan Kiến trúc

Hệ thống được xây dựng dựa trên **Deep Q-Network (DQN)** - một thuật toán Deep Reinforcement Learning để gợi ý sản phẩm thông minh. Model học từ feedback của user (click/purchase) để cải thiện chất lượng recommendation theo thời gian.

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND                                 │
│  (User Interface - 3 positions: Home, Search, Cart)             │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API LAYER (FastAPI)                         │
│  • POST /recommend    → Gợi ý sản phẩm                          │
│  • POST /train        → Training từ feedback                     │
│  • GET  /status       → Trạng thái model                         │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DQN AGENT (Core Logic)                         │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  1. State Encoder: Raw Data → State Vector (87-dim) │       │
│  │  2. DQN Model: Neural Network (3 layers)            │       │
│  │  3. Target Network: Stable Q-value estimation        │       │
│  │  4. Replay Buffer: Experience storage (10K capacity)│       │
│  │  5. Epsilon-Greedy: Exploration vs Exploitation     │       │
│  └──────────────────────────────────────────────────────┘       │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   MODEL PERSISTENCE                              │
│  • dqn_model.pt: Main model (auto-save after each train)       │
│  • checkpoints/: Backup every 100 trains                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Các Thành Phần Chính

### 1. **State Encoder** (`state_encoder.py`)

**Chức năng:** Chuyển đổi dữ liệu thô từ user thành vector số cố định (87 chiều) để đưa vào neural network.

#### Cấu trúc State Vector (87 chiều):

```
┌─────────────────────────────────────────────────────────────┐
│  COMMON FEATURES (15 chiều)                                 │
├─────────────────────────────────────────────────────────────┤
│  • Gender (3): One-hot [Male, Female, Other]                │
│  • Age Group (5): One-hot [U20, U30, U40, U50, U60]        │
│  • Day of Week (7): One-hot [Mon-Sun]                       │
└─────────────────────────────────────────────────────────────┘
                         +
┌─────────────────────────────────────────────────────────────┐
│  POSITION-SPECIFIC FEATURES (69 chiều)                      │
├─────────────────────────────────────────────────────────────┤
│  CART:                                                       │
│    • Num products (1): Normalized [0-20] → [0,1]           │
│    • Total value (1): Normalized [0-20M] → [0,1]           │
│    • Avg value (1): Normalized [0-2M] → [0,1]              │
│    • Product IDs (50): One-hot encoding                     │
│    • Categories (10): Multi-hot encoding                    │
│    • Padding (6): Zeros                                     │
│                                                              │
│  SEARCH:                                                     │
│    • Recent searches (1): Normalized [0-50] → [0,1]        │
│    • Product IDs (50): One-hot encoding                     │
│    • Categories (10): Multi-hot encoding                    │
│    • Padding (8): Zeros                                     │
│                                                              │
│  HOME:                                                       │
│    • Product IDs (50): One-hot (trending/popular)           │
│    • Categories (10): Multi-hot (preferences)               │
│    • Padding (9): Zeros                                     │
└─────────────────────────────────────────────────────────────┘
                         +
┌─────────────────────────────────────────────────────────────┐
│  POSITION ENCODING (3 chiều)                                │
├─────────────────────────────────────────────────────────────┤
│  • Position (3): One-hot [search, cart, home]               │
└─────────────────────────────────────────────────────────────┘
                         =
              TOTAL: 87 DIMENSIONS
```

#### Normalization Strategy:

- **Numeric values:** Min-max normalization về [0, 1]
- **Categorical:** One-hot hoặc Multi-hot encoding
- **Missing values:** Default values (gender="Other", age="U30")

---

### 2. **DQN Model** (`model.py`)

**Kiến trúc Neural Network:**

```
Input Layer (87)
      │
      ▼
┌──────────────────┐
│  FC1 (87 → 128)  │  ← Fully Connected Layer 1
│  + ReLU          │
└──────────────────┘
      │
      ▼
┌──────────────────┐
│  FC2 (128 → 128) │  ← Fully Connected Layer 2
│  + ReLU          │
└──────────────────┘
      │
      ▼
┌──────────────────┐
│  FC3 (128 → 50)  │  ← Output Layer (Q-values)
│  (No activation) │
└──────────────────┘
      │
      ▼
Output: Q-values (50)
[Q(s,1), Q(s,2), ..., Q(s,50)]
```

**Đặc điểm:**

- **Architecture:** 3-layer feedforward network
- **Hidden dimensions:** 128 neurons per layer
- **Activation:** ReLU (Rectified Linear Unit)
- **Output:** Raw Q-values (không có activation ở output)
- **Parameters:** ~20K trainable parameters

---

### 3. **DQN Agent** (`agent.py`)

**Trái tim của hệ thống** - Quản lý toàn bộ logic training và recommendation.

#### Thành phần:

```python
class DQNAgent:
    • model: Main DQN network (online network)
    • target_model: Target DQN network (stable)
    • optimizer: Adam optimizer (lr=0.001)
    • memory: Replay Buffer (capacity=10000)
    • epsilon: Exploration rate (0.5 → 0.3)
    • gamma: Discount factor (0.99)
```

#### Chiến lược Epsilon-Greedy:

```
Epsilon (ε) Timeline:
─────────────────────────────────────────────────
Epoch:    0      500    1000   1500   2000+
Epsilon: 0.5 →  0.4  →  0.35 → 0.32 → 0.30
─────────────────────────────────────────────────
Strategy:
  • ε% → Random action (Exploration)
  • (1-ε)% → Best Q-value action (Exploitation)

Final: 30% exploration, 70% exploitation
```

**Lý do epsilon_min = 0.3:**

- Luôn khám phá 30% sản phẩm mới
- Cân bằng giữa diversity và relevance
- Phù hợp với recommendation system (không cần optimal policy tuyệt đối)

---

### 4. **Replay Buffer** (`replay_buffer.py`)

**Chức năng:** Lưu trữ experiences để training off-policy.

```
┌─────────────────────────────────────────────────┐
│  Replay Buffer (FIFO Queue)                     │
│  Capacity: 10,000 experiences                   │
├─────────────────────────────────────────────────┤
│  Each Experience:                               │
│    (state, action, reward, next_state, done)   │
│                                                  │
│  Example:                                       │
│  ┌────────────────────────────────────────┐    │
│  │ state: [0.5, 0, 1, ..., 0.8]  (87-dim)│    │
│  │ action: 15  (Product ID)               │    │
│  │ reward: 1.0  (Purchase)                │    │
│  │ next_state: [0.5, 0, 1, ..., 0.9]      │    │
│  │ done: False                            │    │
│  └────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘

Operations:
  • push(): Add new experience (auto-remove oldest if full)
  • sample(batch_size): Random sample for training
  • __len__(): Current buffer size
```

**Lợi ích:**

- **Break correlation:** Training trên batch random → stable learning
- **Data efficiency:** Reuse experiences multiple times
- **Online learning:** Liên tục update từ user feedback

---

## 🔄 Flow Hoạt Động Chi Tiết

### Flow 1: **Recommendation Flow** (GET suggestions)

```
┌─────────────┐
│   USER      │
│  on Cart    │
└──────┬──────┘
       │
       │ 1. Request recommendation
       ▼
┌──────────────────────────────────────────┐
│  Frontend gửi raw_data + position        │
│  {                                       │
│    "raw_data": {                         │
│      "gender": "Female",                 │
│      "age_group": "U30",                 │
│      "num_products": 3,                  │
│      "product_ids": [1, 5, 10], ...     │
│    },                                    │
│    "position": "cart"                    │
│  }                                       │
└──────┬───────────────────────────────────┘
       │
       │ 2. POST /recommend
       ▼
┌──────────────────────────────────────────┐
│  API Layer (FastAPI)                     │
│  • Validate input data                   │
│  • Call state_encoder                    │
└──────┬───────────────────────────────────┘
       │
       │ 3. encode_state()
       ▼
┌──────────────────────────────────────────┐
│  State Encoder                           │
│  • Normalize numeric values              │
│  • One-hot encode categoricals           │
│  • Concatenate to 87-dim vector          │
│                                           │
│  Output: state = [0.5, 0, 1, ..., 0.8]  │
└──────┬───────────────────────────────────┘
       │
       │ 4. select_top_actions(state, k=10)
       ▼
┌──────────────────────────────────────────┐
│  DQN Agent - Decision Making             │
│                                           │
│  IF model NOT trained:                   │
│    → Random select 10 products           │
│  ELSE:                                    │
│    Random number r ∈ [0,1]               │
│    IF r < epsilon:                       │
│      → Random select 10 (Exploration)    │
│    ELSE:                                  │
│      → DQN forward pass                  │
│      → Select top 10 Q-values            │
│        (Exploitation)                    │
└──────┬───────────────────────────────────┘
       │
       │ 5. Forward Pass (if exploitation)
       ▼
┌──────────────────────────────────────────┐
│  DQN Model                               │
│  state (87) → FC1 → ReLU → FC2 → ReLU   │
│            → FC3 → Q-values (50)         │
│                                           │
│  Output: [2.3, 1.5, ..., 3.8, ...]      │
│           ↑high Q-value = good product   │
└──────┬───────────────────────────────────┘
       │
       │ 6. Top-K Selection
       ▼
┌──────────────────────────────────────────┐
│  Sort Q-values descending                │
│  Pick top 10 indices                     │
│  Convert: index → Product ID (+1)        │
│                                           │
│  Example: [15, 23, 8, 42, 19, ...]      │
└──────┬───────────────────────────────────┘
       │
       │ 7. Return response
       ▼
┌──────────────────────────────────────────┐
│  API Response                            │
│  {                                       │
│    "recommended_products": [15,23,8,...],│
│    "count": 10,                          │
│    "strategy": "epsilon-greedy (ε=0.35)",│
│    "model_status": "trained"             │
│  }                                       │
└──────┬───────────────────────────────────┘
       │
       │ 8. Display recommendations
       ▼
┌──────────────┐
│  USER sees   │
│  Top 10      │
│  Products    │
└──────────────┘
```

---

### Flow 2: **Training Flow** (Learn from feedback)

```
┌─────────────┐
│   USER      │
│  clicks or  │
│  purchases  │
│  Product 15 │
└──────┬──────┘
       │
       │ 1. User interaction event
       ▼
┌──────────────────────────────────────────┐
│  Frontend tracks interaction             │
│  • Capture: current state, action, result│
│  • Calculate reward:                     │
│    - Click: 0.5                          │
│    - Purchase: 1.0                       │
│  • Capture next state                    │
└──────┬───────────────────────────────────┘
       │
       │ 2. POST /train
       ▼
┌──────────────────────────────────────────┐
│  API Layer                               │
│  Receives:                               │
│  {                                       │
│    "raw_data": {...},  // before         │
│    "position": "cart",                   │
│    "action": 15,       // clicked product│
│    "reward": 1.0,      // purchased!     │
│    "next_raw_data": {...},  // after     │
│    "next_position": "cart",              │
│    "done": false                         │
│  }                                       │
└──────┬───────────────────────────────────┘
       │
       │ 3. Encode states
       ▼
┌──────────────────────────────────────────┐
│  State Encoder (called 2x)               │
│  • state = encode(raw_data, position)    │
│  • next_state = encode(next_raw_data, ...)│
└──────┬───────────────────────────────────┘
       │
       │ 4. Store experience
       ▼
┌──────────────────────────────────────────┐
│  Replay Buffer.push()                    │
│  • Add (s, a, r, s', done) to buffer     │
│  • Current size: 1251 / 10000            │
└──────┬───────────────────────────────────┘
       │
       │ 5. Sample batch for training
       ▼
┌──────────────────────────────────────────┐
│  Replay Buffer.sample(batch_size=32)     │
│  • Random sample 32 experiences          │
│  • Convert to numpy arrays               │
│  • Convert to PyTorch tensors            │
└──────┬───────────────────────────────────┘
       │
       │ 6. Calculate target Q-values
       ▼
┌──────────────────────────────────────────┐
│  Target Network Forward Pass             │
│  FOR each next_state in batch:           │
│    Q_target(next_state) → [50 Q-values]  │
│    max_Q = max(Q_target(next_state))     │
│                                           │
│  Bellman Equation:                       │
│  Q_target_value = r + γ * max_Q * (1-done)│
│                                           │
│  where:                                  │
│    r = reward (0.5 or 1.0)               │
│    γ = 0.99 (discount factor)            │
│    done = False usually                  │
└──────┬───────────────────────────────────┘
       │
       │ 7. Calculate current Q-values
       ▼
┌──────────────────────────────────────────┐
│  Main Network Forward Pass               │
│  FOR each state in batch:                │
│    Q_current(state) → [50 Q-values]      │
│    Q_current[action] = predicted Q-value │
└──────┬───────────────────────────────────┘
       │
       │ 8. Calculate loss
       ▼
┌──────────────────────────────────────────┐
│  Loss Function: MSE Loss                 │
│                                           │
│  Loss = Mean(                            │
│    (Q_current[action] - Q_target_value)² │
│  )                                        │
│                                           │
│  Goal: Make predicted Q-value close to   │
│        actual reward + future reward     │
└──────┬───────────────────────────────────┘
       │
       │ 9. Backpropagation
       ▼
┌──────────────────────────────────────────┐
│  Optimizer (Adam)                        │
│  • optimizer.zero_grad()                 │
│  • loss.backward()  ← compute gradients  │
│  • optimizer.step() ← update weights     │
│                                           │
│  → DQN model weights updated!            │
└──────┬───────────────────────────────────┘
       │
       │ 10. Update epsilon
       ▼
┌──────────────────────────────────────────┐
│  Epsilon Decay                           │
│  epsilon = max(0.3, epsilon * 0.995)     │
│  → Gradually shift to exploitation       │
└──────┬───────────────────────────────────┘
       │
       │ 11. Save model (EVERY train)
       ▼
┌──────────────────────────────────────────┐
│  Model Persistence                       │
│  • Save to dqn_model.pt                  │
│  • Include: model, target, optimizer, ε  │
│                                           │
│  IF train_count % 100 == 0:              │
│    • Update target network               │
│    • Create checkpoint backup            │
└──────┬───────────────────────────────────┘
       │
       │ 12. Return training status
       ▼
┌──────────────────────────────────────────┐
│  API Response                            │
│  {                                       │
│    "status": "trained",                  │
│    "epsilon": 0.348,                     │
│    "memory_size": 1251,                  │
│    "train_count": 1251,                  │
│    "model_activated": true,              │
│    "model_saved": true                   │
│  }                                       │
└──────┬───────────────────────────────────┘
       │
       │ 13. Log & Continue
       ▼
┌──────────────┐
│  System      │
│  ready for   │
│  next request│
└──────────────┘
```

---

## 🧠 Thuật Toán DQN Chi Tiết

### Bellman Equation (Core of Q-Learning):

```
Q(s, a) = r + γ * max[Q(s', a')]
          └─┘   └─┘   └──────────┘
           │     │         │
           │     │         └─ Best future Q-value
           │     └─ Discount factor (0.99)
           └─ Immediate reward

Where:
  s  = current state
  a  = action taken
  r  = reward received
  s' = next state
  γ  = discount factor (how much we value future rewards)
```

## 📊 Training Process (Step by Step)

### Single Training Step:

```python
def train_step(batch_size=32):
    # Step 1: Sample batch from replay buffer
    states, actions, rewards, next_states, dones = memory.sample(32)

    # Step 2: Convert to tensors
    states = torch.FloatTensor(states)          # (32, 87)
    actions = torch.LongTensor(actions)         # (32,)
    rewards = torch.FloatTensor(rewards)        # (32,)
    next_states = torch.FloatTensor(next_states)# (32, 87)
    dones = torch.FloatTensor(dones)            # (32,)

    # Step 3: Calculate target Q-values (no gradient)
    with torch.no_grad():
        q_next = target_model(next_states)      # (32, 50)
        max_q_next = q_next.max(dim=1)[0]       # (32,)
        q_target = rewards + 0.99 * max_q_next * (1 - dones)

    # Step 4: Calculate current Q-values
    q_values = model(states)                    # (32, 50)
    q_current = q_values.gather(1, actions.unsqueeze(1))  # (32, 1)

    # Step 5: Calculate loss
    loss = MSELoss(q_current, q_target)

    # Step 6: Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Step 7: Update epsilon
    epsilon = max(0.3, epsilon * 0.995)
```

### Strategy Evolution:

```
Phase 1: Cold Start (train_count = 0)
├─ Strategy: 100% Random
├─ Model: Not activated
└─ Goal: Collect initial data

Phase 2: Early Training (train_count = 1-200)
├─ Strategy: 50% Random, 50% Model
├─ Epsilon: 0.5 → 0.4
└─ Goal: Learn basic patterns

Phase 3: Mid Training (train_count = 200-1000)
├─ Strategy: 40% Random, 60% Model
├─ Epsilon: 0.4 → 0.32
└─ Goal: Refine recommendations

Phase 4: Mature (train_count > 1000)
├─ Strategy: 30% Random, 70% Model
├─ Epsilon: 0.3 (stable)
└─ Goal: Balance quality & diversity
```

### Recommendation Quality:

```
Early Stage (train_count < 100):
  → Mostly random, diverse but not relevant

Mid Stage (train_count 100-500):
  → Learning patterns, improving relevance

Mature Stage (train_count > 500):
  → High relevance + 30% diversity
```

## 🚀 Deployment Flow

### Complete Lifecycle:

```
1. STARTUP
   ├─ Initialize API (FastAPI)
   ├─ Initialize DQN Agent
   ├─ Check for existing model
   │   ├─ IF dqn_model.pt exists:
   │   │   └─ Load model → Ready immediately
   │   └─ ELSE:
   │       └─ Cold start → Random recommendations
   └─ API ready to serve

2. RECOMMENDATION PHASE
   ├─ User visits page (Home/Search/Cart)
   ├─ Frontend calls POST /recommend
   ├─ API returns top 10 products
   └─ User sees recommendations

3. INTERACTION PHASE
   ├─ User clicks/purchases a product
   ├─ Frontend calls POST /train
   ├─ Model updates (1 training step)
   ├─ Model auto-saves
   └─ Next recommendation will be better

4. CONTINUOUS LEARNING
   ├─ More users → More feedback
   ├─ More training → Better model
   ├─ Epsilon decreases → More exploitation
   └─ Quality improves over time

5. RESTART (Safe)
   ├─ Stop API
   ├─ Model already saved
   ├─ Restart API
   ├─ Auto-load model
   └─ Continue from where it stopped
```

---

## 🎓 Ưu Điểm của Kiến Trúc

### 1. **Online Learning**

- Học liên tục từ user feedback real-time
- Không cần offline training phase
- Model cải thiện theo thời gian

### 2. **Personalized Context**

- State vector chứa đầy đủ context (user, position, products)
- 3 vị trí khác nhau → 3 contexts khác nhau
- Model học pattern riêng cho từng context

### 3. **Exploration vs Exploitation**

- Epsilon-greedy đảm bảo diversity
- 30% exploration → Khám phá sản phẩm mới
- 70% exploitation → Gợi ý chất lượng cao

### 4. **Stable Training**

- Replay buffer breaks correlation
- Target network prevents moving target
- Gradual epsilon decay ensures convergence

### 5. **Fault Tolerance**

- Auto-save sau mỗi train
- Checkpoint backups
- Restart-safe (load from file)
