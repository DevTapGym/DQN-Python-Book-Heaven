# DQN Recommendation API - Hướng dẫn sử dụng

## 🚀 Khởi động API

### Cài đặt dependencies:

```bash
cd d:\University\Graduation_Project\AI\DQN\common
pip install -r requirements.txt
```

### Chạy API server:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

**API URL:** http://localhost:8000  
**Interactive Docs:** http://localhost:8000/docs

---

## 📋 Danh sách Endpoints

| Method | Endpoint         | Mô tả                      |
| ------ | ---------------- | -------------------------- |
| GET    | `/status`        | Kiểm tra trạng thái agent  |
| POST   | `/recommend`     | Gợi ý sản phẩm             |
| POST   | `/train`         | Training model từ feedback |
| POST   | `/update_target` | Cập nhật target network    |
| POST   | `/save_model`    | Lưu model checkpoint       |
| POST   | `/load_model`    | Load model từ checkpoint   |

---

## 📖 Chi tiết từng Endpoint

### 1. GET /status

**Mô tả:** Lấy thông tin trạng thái hiện tại của agent

**Request:**

```http
GET http://localhost:8000/status
```

**Response:**

```json
{
  "epsilon": 0.85,
  "memory_size": 120,
  "state_dim": 85,
  "action_dim": 50
}
```

**Giải thích:**

- `epsilon`: Tỷ lệ exploration (1.0 → 0.1, giảm dần qua training)
- `memory_size`: Số experiences đã lưu trong replay buffer (max: 5000)
- `state_dim`: Kích thước state vector (87 chiều)
- `action_dim`: Số sản phẩm có thể gợi ý (50 sản phẩm, ID từ 1-50)

---

### 2. POST /recommend

**Mô tả:** Gợi ý **top 10 sản phẩm** dựa trên context của user

#### 🛒 Case 1: Gợi ý trong Cart (Người dùng đang có sản phẩm trong giỏ)

**Khi nào dùng:** User đang xem giỏ hàng, cần gợi ý thêm sản phẩm liên quan

**Request:**

```json
POST http://localhost:8000/recommend
Content-Type: application/json

{
  "raw_data": {
    "gender": "Female",
    "age_group": "U30",
    "day_of_week": 3,
    "num_products": 3,
    "total_value": 500000,
    "avg_value": 166666,
    "product_ids": [1, 5, 10],
    "category": ["Music", "Travel"]
  },
  "position": "cart"
}
```

**Response:**

```json
{
  "recommended_products": [15, 23, 8, 42, 19, 31, 7, 28, 12, 45],
  "count": 10
}
```

**Giải thích:**

- `recommended_products`: Danh sách 10 product IDs (1-50) được xếp theo độ ưu tiên từ cao đến thấp
- `count`: Số lượng sản phẩm gợi ý (luôn là 10)

---

#### 🔍 Case 2: Gợi ý tại Search (Người dùng đang tìm kiếm)

**Khi nào dùng:** User vừa search, cần gợi ý sản phẩm phù hợp với lịch sử tìm kiếm

**Request:**

```json
{
  "raw_data": {
    "gender": "Male",
    "age_group": "U40",
    "day_of_week": 5,
    "recent_searches": 8,
    "product_ids": [12, 15, 23],
    "category": ["Business", "History"]
  },
  "position": "search"
}
```

**Response:**

```json
{
  "recommended_products": [23, 15, 38, 11, 29, 6, 41, 18, 33, 9],
  "count": 10
}
```

**Giải thích:**

- `recent_searches`: Số lượng tìm kiếm gần đây (0-50)
- `product_ids`: Các sản phẩm đã xem trong phiên tìm kiếm
- `category`: Các danh mục đã tìm kiếm

---

#### 🏠 Case 3: Gợi ý tại Home (Trang chủ - chưa có context cụ thể)

**Khi nào dùng:** User vừa vào trang chủ, chưa có hành động cụ thể

**Request:**

```json
{
  "raw_data": {
    "gender": "Female",
    "age_group": "U20",
    "day_of_week": 1,
    "product_ids": [5, 12, 18, 25],
    "category": ["Business", "Entertainment"]
  },
  "position": "home"
}
```

**Response:**

```json
{
  "recommended_products": [32, 14, 27, 8, 43, 19, 36, 22, 11, 48],
  "count": 10
}
```

**Giải thích:**

- `product_ids`: Top sản phẩm phổ biến hoặc trending
- `category`: Danh mục phổ biến hoặc preferences của user

---

### 3. POST /train

**Mô tả:** Training model từ feedback của user khi có tương tác với sản phẩm gợi ý

**Khi nào gọi:**

- User CLICK vào sản phẩm gợi ý → reward = 0.5
- User MUA sản phẩm gợi ý → reward = 1.0
- KHÔNG gọi nếu user không tương tác gì

**Request Example - User mua sản phẩm ở Cart:**

```json
POST http://localhost:8000/train
Content-Type: application/json

{
  "raw_data": {
    "gender": "Female",
    "age_group": "U30",
    "day_of_week": 3,
    "num_products": 3,
    "total_value": 500000,
    "avg_value": 166666,
    "product_ids": [1, 5, 10],
    "category": ["Music", "Travel"]
  },
  "position": "cart",
  "action": 15,
  "reward": 1.0,
  "next_raw_data": {
    "gender": "Female",
    "age_group": "U30",
    "day_of_week": 3,
    "num_products": 4,
    "total_value": 750000,
    "avg_value": 187500,
    "product_ids": [1, 5, 10, 15],
    "category": ["Music", "Travel", "Arts"]
  },
  "next_position": "cart",
  "done": false
}
```

**Reward Values:**
| Hành động | Reward | Ghi chú |
|-----------|--------|---------|
| Không tương tác | Không gửi | Skip training |
| Click | 0.5 | User xem sản phẩm |
| Purchase | 1.0 | User mua sản phẩm |

**Lưu ý:**

- `action`: Phải là product ID từ danh sách `recommended_products` (1-50)
- `next_raw_data`: State sau khi user tương tác (có thể thay đổi hoặc giữ nguyên)
- `done`: Luôn là `false` (trừ khi user checkout hoàn tất session)

**Response:**

```json
{
  "status": "trained",
  "epsilon": 0.84
}
```

---

### 4. POST /update_target

**Mô tả:** Đồng bộ target network với main network

**Khi nào gọi:** Mỗi 100-200 training steps

**Request:**

```http
POST http://localhost:8000/update_target
```

**Response:**

```json
{
  "status": "target_updated"
}
```

---

### 5. POST /save_model

**Mô tả:** Lưu model checkpoint

**Request:**

```http
POST http://localhost:8000/save_model?path=dqn_checkpoint_v1.pt
```

**Response:**

```json
{
  "status": "saved",
  "path": "dqn_checkpoint_v1.pt"
}
```

**Nội dung checkpoint:**

- Model weights
- Target model weights
- Optimizer state
- Epsilon value

---

### 6. POST /load_model

**Mô tả:** Load model từ checkpoint

**Request:**

```http
POST http://localhost:8000/load_model?path=dqn_checkpoint_v1.pt
```

**Response:**

```json
{
  "status": "loaded",
  "path": "dqn_checkpoint_v1.pt"
}
```

---

## 🎯 Use Cases - 3 Kịch bản thực tế

### Case 1: User vào Cart và mua sản phẩm được gợi ý

```javascript
// 1. User vào cart, có 3 sản phẩm
const cartState = {
  gender: "Female",
  age_group: "U30",
  day_of_week: 3,
  num_products: 3,
  total_value: 500000,
  avg_value: 166666,
  product_ids: [1, 5, 10],
  category: ["Music", "Travel"],
};

// 2. Gọi API gợi ý top 10 sản phẩm
const response = await fetch("http://localhost:8000/recommend", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    raw_data: cartState,
    position: "cart",
  }),
});
const { recommended_products } = await response.json();
// recommended_products = [15, 23, 8, 42, 19, 31, 7, 28, 12, 45]

// 3. Hiển thị 10 sản phẩm cho user
// 4. User CLICK và MUA sản phẩm 15 (sản phẩm đầu tiên)

// 5. Gửi feedback training với reward = 1.0
const nextState = {
  ...cartState,
  num_products: 4,
  total_value: 750000,
  avg_value: 187500,
  product_ids: [1, 5, 10, 15],
  category: ["Music", "Travel", "Arts"],
};

await fetch("http://localhost:8000/train", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    raw_data: cartState,
    position: "cart",
    action: 15, // Sản phẩm user đã mua
    reward: 1.0, // Purchase = reward cao
    next_raw_data: nextState,
    next_position: "cart",
    done: false,
  }),
});
```

---

### Case 2: User search và chỉ click xem sản phẩm (không mua)

```javascript
// 1. User đang search, có 8 lần tìm kiếm gần đây
const searchState = {
  gender: "Male",
  age_group: "U40",
  day_of_week: 5,
  recent_searches: 8,
  product_ids: [12, 15, 23],
  category: ["Business", "History"],
};

// 2. Gọi API gợi ý top 10
const response = await fetch("http://localhost:8000/recommend", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    raw_data: searchState,
    position: "search",
  }),
});
const { recommended_products } = await response.json();
// recommended_products = [23, 15, 38, 11, 29, 6, 41, 18, 33, 9]

// 3. User CLICK vào sản phẩm 23 để xem chi tiết
// 4. User KHÔNG MUA, tiếp tục search

// 5. Gửi feedback với reward = 0.5 (chỉ click)
const nextState = {
  ...searchState,
  recent_searches: 9, // Tăng số lần search
  product_ids: [12, 15, 23, 38], // Thêm sản phẩm đã xem
};

await fetch("http://localhost:8000/train", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    raw_data: searchState,
    position: "search",
    action: 23, // Sản phẩm user đã click
    reward: 0.5, // Click only = reward thấp hơn
    next_raw_data: nextState,
    next_position: "search",
    done: false,
  }),
});
```

---

### Case 3: User vào Home, xem gợi ý nhưng KHÔNG tương tác

```javascript
// 1. User vào trang chủ
const homeState = {
  gender: "Female",
  age_group: "U20",
  day_of_week: 1,
  product_ids: [5, 12, 18, 25], // Top trending products
  category: ["Business", "Entertainment"],
};

// 2. Gọi API gợi ý top 10
const response = await fetch("http://localhost:8000/recommend", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    raw_data: homeState,
    position: "home",
  }),
});
const { recommended_products } = await response.json();
// recommended_products = [32, 14, 27, 8, 43, 19, 36, 22, 11, 48]

// 3. Hiển thị 10 sản phẩm
// 4. User KHÔNG click vào bất kỳ sản phẩm nào

// 5. KHÔNG gọi /train
// Model sẽ tự học từ feedback của users khác
// Không có negative reward cho trường hợp này
```

---

## 🔄 Workflow tự động

### Cập nhật Target Network định kỳ

```python
import requests

train_count = 0
UPDATE_TARGET_INTERVAL = 100

def on_user_action(state, action, reward, next_state):
    global train_count

    # Train
    requests.post('http://localhost:8000/train', json={
        "raw_data": state,
        "position": "cart",
        "action": action,
        "reward": reward,
        "next_raw_data": next_state,
        "next_position": "cart",
        "done": False
    })

    train_count += 1

    # Update target network định kỳ
    if train_count % UPDATE_TARGET_INTERVAL == 0:
        requests.post('http://localhost:8000/update_target')
        print(f"✅ Target network updated at step {train_count}")
```

---

### Lưu model tự động

```python
import schedule
import requests
from datetime import datetime

def save_checkpoint():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"dqn_model_{timestamp}.pt"
    requests.post(f'http://localhost:8000/save_model?path={path}')
    print(f"✅ Model saved: {path}")

# Lưu mỗi 1 giờ
schedule.every(1).hours.do(save_checkpoint)

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

### Monitoring training progress

```python
import requests
import time

def monitor_training():
    while True:
        response = requests.get('http://localhost:8000/status')
        status = response.json()

        print(f"\n{'='*50}")
        print(f"Epsilon: {status['epsilon']:.3f}")
        print(f"Memory: {status['memory_size']}/5000")
        print(f"Progress: {(status['memory_size']/5000)*100:.1f}%")

        if status['epsilon'] < 0.2 and status['memory_size'] > 500:
            print("✅ Model trained well - Ready for production!")
        elif status['memory_size'] < 500:
            print("⚠️  Need more training data")

        time.sleep(300)  # Check mỗi 5 phút

monitor_training()
```

---

## 📊 Raw Data Schema

### Common Fields (Tất cả positions)

| Field         | Type    | Values                            | Required |
| ------------- | ------- | --------------------------------- | -------- |
| `gender`      | string  | "Male", "Female", "Other"         | ✅       |
| `age_group`   | string  | "U20", "U30", "U40", "U50", "U60" | ✅       |
| `day_of_week` | integer | 1-7 (1=CN, 2=T2, ..., 7=T7)       | ✅       |

---

### Cart Position Fields

| Field          | Type          | Description                  | Required |
| -------------- | ------------- | ---------------------------- | -------- |
| `num_products` | integer       | Số sản phẩm trong giỏ (0-20) | ✅       |
| `total_value`  | integer       | Tổng giá trị giỏ (VNĐ)       | ✅       |
| `avg_value`    | float         | Giá trị trung bình/sản phẩm  | ✅       |
| `products`     | array[int]    | Danh sách ID sản phẩm [1-50] | ✅       |
| `category`     | array[string] | Danh sách thể loại           | ✅       |

**Categories:** "Business", "Entertainment", "Cooking", "History", "Music", "Comics", "Travel", "Arts", "Sports", "Psychology"

---

### Search Position Fields

| Field             | Type    | Description                  | Required |
| ----------------- | ------- | ---------------------------- | -------- |
| `recent_searches` | integer | Số lần search gần đây (0-50) | ✅       |

---

### Home Position Fields

| Field         | Type          | Description                  | Required |
| ------------- | ------------- | ---------------------------- | -------- |
| `product_ids` | array[int]    | Top sản phẩm trending [1-50] | ✅       |
| `category`    | array[string] | Top thể loại trending        | ✅       |

---

## 🧪 Test với cURL (Windows CMD)

### Test recommend - Cart position

```bash
curl -X POST http://localhost:8000/recommend -H "Content-Type: application/json" -d "{\"raw_data\": {\"gender\": \"Female\", \"age_group\": \"U30\", \"day_of_week\": 3, \"num_products\": 3, \"total_value\": 500000, \"avg_value\": 166666, \"product_ids\": [1, 5, 10], \"category\": [\"Music\", \"Travel\"]}, \"position\": \"cart\"}"
```

### Test recommend - Search position

```bash
curl -X POST http://localhost:8000/recommend -H "Content-Type: application/json" -d "{\"raw_data\": {\"gender\": \"Male\", \"age_group\": \"U40\", \"day_of_week\": 5, \"recent_searches\": 8, \"product_ids\": [12, 15, 23], \"category\": [\"Business\", \"History\"]}, \"position\": \"search\"}"
```

### Test recommend - Home position

```bash
curl -X POST http://localhost:8000/recommend -H "Content-Type: application/json" -d "{\"raw_data\": {\"gender\": \"Female\", \"age_group\": \"U20\", \"day_of_week\": 1, \"product_ids\": [5, 12, 18, 25], \"category\": [\"Business\", \"Entertainment\"]}, \"position\": \"home\"}"
```

### Test status

```bash
curl http://localhost:8000/status
```

### Test training

```bash
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d "{\"raw_data\": {\"gender\": \"Female\", \"age_group\": \"U30\", \"day_of_week\": 3, \"num_products\": 3, \"total_value\": 500000, \"avg_value\": 166666, \"product_ids\": [1,5,10], \"category\": [\"Music\"]}, \"position\": \"cart\", \"action\": 15, \"reward\": 1.0, \"next_raw_data\": {\"gender\": \"Female\", \"age_group\": \"U30\", \"day_of_week\": 3, \"num_products\": 4, \"total_value\": 750000, \"avg_value\": 187500, \"product_ids\": [1,5,10,15], \"category\": [\"Music\",\"Arts\"]}, \"next_position\": \"cart\", \"done\": false}"
```

### Test update target

```bash
curl -X POST http://localhost:8000/update_target
```

### Test save model

```bash
curl -X POST "http://localhost:8000/save_model?path=dqn_model_v1.pt"
```

### Test load model

```bash
curl -X POST "http://localhost:8000/load_model?path=dqn_model_v1.pt"
```

---

## ⚠️ Lưu ý quan trọng

### Validation Rules

#### Product IDs

- ✅ Phải từ **1-50** (không phải 0-49)
- ❌ `product_ids: [0, 15, 20]` → Lỗi validation
- ✅ `product_ids: [1, 15, 20]` → OK

#### Position

- ✅ Chỉ chấp nhận: `"search"`, `"cart"`, `"home"`
- ❌ `position: "checkout"` → Lỗi validation

#### Gender & Age

- ✅ `gender`: "Male", "Female", "Other"
- ✅ `age_group`: "U20", "U30", "U40", "U50", "U60"

#### Day of Week

- ✅ Phải là số nguyên từ 1-7 (Monday=1, Sunday=7)

#### Categories

- ✅ Phải nằm trong: "Business", "Entertainment", "Cooking", "History", "Music", "Comics", "Travel", "Arts", "Sports", "Psychology"

#### Numeric Fields

- ✅ `num_products`, `total_value`, `avg_value`, `recent_searches` phải >= 0

---

### Training Requirements

- **Minimum experiences:** 32 (batch size) - Model bắt đầu train
- **Recommended:** 500+ experiences - Model học tốt
- **Optimal:** 5000 experiences - Replay buffer đầy

### Epsilon Schedule

- **Start:** ε = 1.0 (100% exploration)
- **Decay:** 0.995 mỗi training step
- **End:** ε = 0.1 (90% exploitation, 10% exploration)

### Update Frequency

- **Target network:** Mỗi 100 training steps
- **Save model:** Mỗi 1 giờ hoặc 1000 steps
- **Monitor status:** Mỗi 5 phút

### State Normalization

Tất cả continuous values được chuẩn hóa về [0, 1]:

- `num_products`: max = 20
- `total_value`: max = 20,000,000 VNĐ
- `avg_value`: max = 2,000,000 VNĐ
- `recent_searches`: max = 50

### Best Practices

1. ✅ Luôn gửi feedback khi user tương tác
2. ✅ Update target network định kỳ
3. ✅ Lưu model checkpoint thường xuyên
4. ✅ Monitor epsilon và memory_size
5. ✅ Đảm bảo `next_raw_data` là state SAU KHI user tương tác
6. ❌ KHÔNG gửi feedback khi user không tương tác

---

## 📈 Training Progress

### Phase 1: Initial Training (Episodes 0-100)

- Epsilon: 1.0 → 0.6
- Memory: 0 → 100
- Behavior: Mostly random recommendations

### Phase 2: Learning (Episodes 100-500)

- Epsilon: 0.6 → 0.18
- Memory: 100 → 500
- Behavior: Starting to learn patterns

### Phase 3: Optimization (Episodes 500-1000)

- Epsilon: 0.18 → 0.1
- Memory: 500 → 5000 (full)
- Behavior: Well-trained, mostly exploiting

### Phase 4: Production Ready (Episodes 1000+)

- Epsilon: 0.1 (stable)
- Memory: 5000 (full, FIFO)
- Behavior: 90% learned, 10% exploration

---

## 🔧 Troubleshooting

### Model không học?

- ✅ Check memory_size >= 32
- ✅ Check epsilon đang giảm
- ✅ Verify reward values đúng
- ✅ Ensure đang gửi đúng next_state

### Recommendations không đổi?

- ✅ Check epsilon (nếu = 1.0, đang full exploration)
- ✅ Cần thêm training data
- ✅ Update target network

### Training chậm?

- ✅ Reduce batch_size (trong agent.py)
- ✅ Tăng UPDATE_TARGET_INTERVAL
- ✅ Check GPU availability

---

## 📞 Support

Để biết thêm chi tiết về thuật toán và implementation:

- Xem file `agent.py` - DQN Agent implementation
- Xem file `model.py` - Neural network architecture
- Xem file `state_encoder.py` - State encoding logic

---

**🎯 Ready to use! Start sending requests to the API!**
