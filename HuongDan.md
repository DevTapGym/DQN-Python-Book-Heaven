# DQN Recommendation API - Hướng dẫn sử dụng

## 🚀 Khởi động API

### Cài đặt dependencies:

```bash
pip install -r requirements.txt
```

### Chạy API server:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### ✅ Tự động load model:

- Khi khởi động, API sẽ tự động load model từ `dqn_model.pt` nếu file tồn tại
- Nếu chưa có model → bắt đầu từ đầu (cold start)
- Model được lưu **sau mỗi lần train**, không lo mất dữ liệu

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
  "epsilon": 0.35,
  "memory_size": 120,
  "train_count": 856,
  "model_activated": true,
  "state_dim": 87,
  "action_dim": 50,
  "strategy": "epsilon-greedy (ε=0.35)"
}
```

**Giải thích:**

- `epsilon`: Tỷ lệ exploration (0.5 → 0.3, giảm dần qua training)
- `memory_size`: Số experiences đã lưu trong replay buffer (max: 10000)
- `train_count`: Số lần đã train model
- `model_activated`: Model đã được train chưa (true = đã có model)
- `state_dim`: Kích thước state vector (87 chiều)
- `action_dim`: Số sản phẩm có thể gợi ý (50 sản phẩm, ID từ 1-50)
- `strategy`: Chiến lược hiện tại (random hoặc epsilon-greedy)

---

## 🎯 Cách sử dụng API cho 3 vị trí

### 2. POST /recommend - Gợi ý sản phẩm

**Mô tả:** Gợi ý **top 10 sản phẩm** dựa trên context của user

**Chiến lược:**

- **Chưa có model:** Random 100%
- **Có model:** Epsilon-greedy (30% random khám phá, 70% dùng model)
- Epsilon giảm dần từ 50% → 30% để cân bằng exploration/exploitation

---

#### 🛒 VỊ TRÍ 1: Cart (Giỏ hàng)

**Khi nào dùng:** User đang xem giỏ hàng, cần gợi ý thêm sản phẩm liên quan

**📋 Dữ liệu cần truyền:**

| Field          | Type          | Bắt buộc | Giá trị hợp lệ                    | Mô tả                             |
| -------------- | ------------- | -------- | --------------------------------- | --------------------------------- |
| `gender`       | string        | ✅       | "Male", "Female", "Other"         | Giới tính user                    |
| `age_group`    | string        | ✅       | "U20", "U30", "U40", "U50", "U60" | Nhóm tuổi                         |
| `day_of_week`  | integer       | ✅       | 1-7 (1=Thứ 2, 7=Chủ nhật)         | Ngày trong tuần                   |
| `num_products` | integer       | ✅       | ≥ 0                               | Số sản phẩm trong giỏ             |
| `total_value`  | number        | ✅       | ≥ 0                               | Tổng giá trị giỏ hàng (VNĐ)       |
| `avg_value`    | number        | ✅       | ≥ 0                               | Giá trị trung bình/sản phẩm (VNĐ) |
| `product_ids`  | array[int]    | ✅       | [1-50]                            | Danh sách ID sản phẩm trong giỏ   |
| `category`     | array[string] | ✅       | Xem danh sách bên dưới            | Danh mục sản phẩm                 |
| `position`     | string        | ✅       | "cart"                            | Vị trí gọi API                    |

**📌 Danh mục hợp lệ (categories):**

```
"Business", "Entertainment", "Cooking", "History", "Music",
"Comics", "Travel", "Arts", "Sports", "Psychology"
```

**✅ Validation Rules:**

- `product_ids` phải nằm trong khoảng 1-50
- `num_products` phải khớp với độ dài của `product_ids`
- `avg_value` = `total_value` / `num_products` (nếu num_products > 0)
- Nếu giỏ rỗng: `num_products=0`, `total_value=0`, `avg_value=0`, `product_ids=[]`

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
  "count": 10,
  "strategy": "epsilon-greedy (ε=0.35)",
  "model_status": "trained"
}
```

---

#### 🔍 VỊ TRÍ 2: Search (Tìm kiếm)

**Khi nào dùng:** User vừa search hoặc đang xem kết quả tìm kiếm

**📋 Dữ liệu cần truyền:**

| Field             | Type          | Bắt buộc | Giá trị hợp lệ                    | Mô tả                         |
| ----------------- | ------------- | -------- | --------------------------------- | ----------------------------- |
| `gender`          | string        | ✅       | "Male", "Female", "Other"         | Giới tính user                |
| `age_group`       | string        | ✅       | "U20", "U30", "U40", "U50", "U60" | Nhóm tuổi                     |
| `day_of_week`     | integer       | ✅       | 1-7                               | Ngày trong tuần               |
| `recent_searches` | integer       | ✅       | 0-50                              | Số lần search gần đây         |
| `product_ids`     | array[int]    | ✅       | [1-50]                            | Sản phẩm đã xem trong session |
| `category`        | array[string] | ✅       | Xem danh sách categories          | Danh mục đã tìm kiếm          |
| `position`        | string        | ✅       | "search"                          | Vị trí gọi API                |

**✅ Validation Rules:**

- `recent_searches` không được vượt quá 50
- `product_ids` có thể rỗng nếu user chưa xem sản phẩm nào
- `category` nên chứa các danh mục liên quan đến keyword search
- Nếu mới vào search: `recent_searches=0`, `product_ids=[]`

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
  "count": 10,
  "strategy": "epsilon-greedy (ε=0.35)",
  "model_status": "trained"
}
```

---

#### 🏠 VỊ TRÍ 3: Home (Trang chủ)

**Khi nào dùng:** User vừa vào trang chủ, chưa có hành động cụ thể

**📋 Dữ liệu cần truyền:**

| Field         | Type          | Bắt buộc | Giá trị hợp lệ                    | Mô tả                         |
| ------------- | ------------- | -------- | --------------------------------- | ----------------------------- |
| `gender`      | string        | ✅       | "Male", "Female", "Other"         | Giới tính user                |
| `age_group`   | string        | ✅       | "U20", "U30", "U40", "U50", "U60" | Nhóm tuổi                     |
| `day_of_week` | integer       | ✅       | 1-7                               | Ngày trong tuần               |
| `product_ids` | array[int]    | ✅       | [1-50]                            | Top trending/popular products |
| `category`    | array[string] | ✅       | Xem danh sách categories          | Danh mục phổ biến/preferences |
| `position`    | string        | ✅       | "home"                            | Vị trí gọi API                |

**✅ Validation Rules:**

- `product_ids` nên chứa 5-10 sản phẩm trending hiện tại
- `category` có thể dựa trên:
  - Preferences của user (nếu đã đăng nhập)
  - Top categories phổ biến (nếu user mới)
- User mới/chưa đăng nhập: dùng giá trị mặc định (gender="Other", age_group="U30")

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
  "count": 10,
  "strategy": "epsilon-greedy (ε=0.35)",
  "model_status": "trained"
}
```

---

## 🎓 Training Model

### 3. POST /train

**Mô tả:** Training model từ feedback của user khi có tương tác với sản phẩm gợi ý

**Khi nào gọi:**

- User **CLICK** vào sản phẩm gợi ý → `reward = 0.5`
- User **MUA** sản phẩm gợi ý → `reward = 1.0`
- User **BỎ QUA** (không tương tác) → KHÔNG gọi API

**📋 Dữ liệu cần truyền:**

| Field           | Type    | Bắt buộc | Giá trị hợp lệ              | Mô tả                                  |
| --------------- | ------- | -------- | --------------------------- | -------------------------------------- |
| `raw_data`      | object  | ✅       | Xem phần position tương ứng | State hiện tại (trước khi tương tác)   |
| `position`      | string  | ✅       | "cart", "search", "home"    | Vị trí xảy ra tương tác                |
| `action`        | integer | ✅       | 1-50                        | Product ID user đã click/mua           |
| `reward`        | float   | ✅       | 0.5 hoặc 1.0                | Reward value (click=0.5, purchase=1.0) |
| `next_raw_data` | object  | ✅       | Xem phần position tương ứng | State sau khi tương tác                |
| `next_position` | string  | ✅       | "cart", "search", "home"    | Vị trí sau khi tương tác               |
| `done`          | boolean | ✅       | true/false                  | Session kết thúc? (thường là false)    |

**✅ Validation Rules:**

- `action` PHẢI nằm trong danh sách `recommended_products` vừa trả về từ `/recommend`
- `reward` chỉ có 2 giá trị: **0.5** (click) hoặc **1.0** (purchase)
- `raw_data` và `next_raw_data` phải tuân theo validation của position tương ứng
- `next_position` có thể khác `position` (ví dụ: từ "search" → "cart")
- `done = true` chỉ khi user checkout hoàn tất hoặc đóng session

**⚠️ LƯU Ý QUAN TRỌNG:**

- Model được **lưu tự động sau mỗi lần train** vào file `dqn_model.pt`
- Backup được tạo mỗi 100 lần train vào thư mục `checkpoints/`
- Target network được update mỗi 100 lần train
- Epsilon giảm dần từ 0.5 → 0.3 (dừng ở 30% exploration)

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
  "epsilon": 0.35,
  "memory_size": 1250,
  "train_count": 1250,
  "model_activated": true,
  "model_saved": true
}
```

**Giải thích fields:**

- `status`: Trạng thái training ("trained")
- `epsilon`: Giá trị epsilon hiện tại (giảm dần 0.5 → 0.3)
- `memory_size`: Số experience trong replay buffer
- `train_count`: Tổng số lần đã train
- `model_activated`: Model đã sẵn sàng sử dụng
- `model_saved`: Model đã được lưu vào file

---
