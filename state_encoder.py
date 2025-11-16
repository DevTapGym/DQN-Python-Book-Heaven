# state_encoder.py
import numpy as np
from config import (
    GENDER_MAP, AGE_MAP, CATEGORIES, MAX_PRODUCTS, POSITION_MAP,
    NUM_PRODUCTS_MAX, TOTAL_VALUE_MIN, TOTAL_VALUE_MAX,
    AVG_VALUE_MIN, AVG_VALUE_MAX, RECENT_SEARCHES_MAX
)


def normalize_value(value, min_val, max_val):
    """
    Chuẩn hóa giá trị về khoảng [0, 1]
    Clip để đảm bảo không vượt quá boundaries
    """
    if max_val == min_val:
        return 0.0
    normalized = (value - min_val) / (max_val - min_val)
    return float(np.clip(normalized, 0.0, 1.0))

def encode_state(raw_data: dict, position: str):
    """
    Encode dữ liệu thô thành state vector cố định để feed DQN.
    
    - Các trường trùng lặp: gender, age, day_of_week → luôn đầu vector
    - Các feature đặc thù theo vị trí nối phía sau
    - Position cuối vector
    """
    state_vec = []

    # --- Các trường trùng lặp (dùng chung cho tất cả vị trí) ---
    state_vec.extend(GENDER_MAP.get(raw_data.get("gender", "Other"), [0,0,1]))
    state_vec.extend(AGE_MAP.get(raw_data.get("age_group", "U20"), [1,0,0,0,0]))
    
    day_vec = np.zeros(7)
    day_of_week = raw_data.get("day_of_week", 1)
    if 1 <= day_of_week <= 7:
        day_vec[day_of_week-1] = 1
    state_vec.extend(day_vec)

    # --- Các feature đặc thù theo vị trí ---
    pos = position.lower()
    if pos == "search":
        # Recent searches - CHUẨN HÓA
        recent_searches = raw_data.get("recent_searches", 0)
        state_vec.append(normalize_value(recent_searches, 0, RECENT_SEARCHES_MAX))
        # Keyword placeholder
        state_vec.extend([0]*5)
        # Products & categories padding (không dùng)
        state_vec.extend([0]*MAX_PRODUCTS)
        state_vec.extend([0]*len(CATEGORIES))
        # Total/avg padding
        state_vec.extend([0,0,0])

    elif pos == "cart":
        # Number of products - CHUẨN HÓA
        num_products = raw_data.get("num_products", 0)
        state_vec.append(normalize_value(num_products, 0, NUM_PRODUCTS_MAX))
        
        # Total value - CHUẨN HÓA
        total_value = raw_data.get("total_value", 0)
        state_vec.append(normalize_value(total_value, TOTAL_VALUE_MIN, TOTAL_VALUE_MAX))
        
        # Average value - CHUẨN HÓA
        avg_value = raw_data.get("avg_value", 0)
        state_vec.append(normalize_value(avg_value, AVG_VALUE_MIN, AVG_VALUE_MAX))
        
        # Products one-hot
        product_vec = np.zeros(MAX_PRODUCTS)
        for pid in raw_data.get("products", []):
            if 1 <= pid <= MAX_PRODUCTS:
                product_vec[pid-1] = 1
        state_vec.extend(product_vec)
        # Categories one-hot (mảng)
        cat_vec = np.zeros(len(CATEGORIES))
        for cat in raw_data.get("category", []):
            if cat in CATEGORIES:
                cat_vec[CATEGORIES.index(cat)] = 1
        state_vec.extend(cat_vec)
        # Keyword placeholder
        state_vec.extend([0]*5)
        # Padding để đồng nhất với search (69 chiều)
        state_vec.append(0)

    elif pos == "home":
        # Top products one-hot
        product_vec = np.zeros(MAX_PRODUCTS)
        for pid in raw_data.get("top_products", []):
            if 1 <= pid <= MAX_PRODUCTS:
                product_vec[pid-1] = 1
        state_vec.extend(product_vec)
        # Top categories one-hot (mảng)
        cat_vec = np.zeros(len(CATEGORIES))
        for cat in raw_data.get("top_categories", []):
            if cat in CATEGORIES:
                cat_vec[CATEGORIES.index(cat)] = 1
        state_vec.extend(cat_vec)
        # Number of products, total/avg, keyword padding
        state_vec.extend([0,0,0])
        state_vec.extend([0]*5)

    else:
        raise ValueError("Position must be one of 'search','cart','home'")
    
    # Position one-hot encoding (3 chiều) - Đặt ở cuối để đồng nhất
    position_vec = np.zeros(3)
    position_vec[POSITION_MAP[pos]] = 1
    state_vec.extend(position_vec)

    result = np.array(state_vec, dtype=np.float32)
    
    # Debug: In ra số chiều để kiểm tra
    if len(result) != 87:
        print(f"WARNING: State dimension mismatch! Expected 87, got {len(result)} for position '{pos}'")
        print(f"State breakdown:")
        print(f"  Common features: gender(3) + age(5) + day(7) = 15")
        print(f"  Position-specific features: {len(result) - 15 - 3}")
        print(f"  Position encoding: 3")
        print(f"  Total: {len(result)}")
    
    return result
