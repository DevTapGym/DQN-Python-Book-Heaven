# config.py
# Các tham số chuẩn hóa cho DQN state encoder

# Boundaries cho normalization về [0, 1]

# Cart-related
NUM_PRODUCTS_MAX = 20           # Số lượng sản phẩm tối đa trong giỏ
TOTAL_VALUE_MIN = 0
TOTAL_VALUE_MAX = 20000000      # 10 triệu VNĐ
AVG_VALUE_MIN = 0
AVG_VALUE_MAX = 2000000         # 2 triệu VNĐ/sản phẩm

# Search-related
RECENT_SEARCHES_MAX = 50        # Số lượng tìm kiếm gần đây tối đa

# Categories
CATEGORIES = [
    "Business", "Entertainment", "Cooking", "History", "Music",
    "Comics", "Travel", "Arts", "Sports", "Psychology"
]

# Gender mapping
GENDER_MAP = {"Male": [1,0,0], "Female": [0,1,0], "Other": [0,0,1]}

# Age group mapping
AGE_MAP = {
    "U20": [1,0,0,0,0], 
    "U30": [0,1,0,0,0], 
    "U40": [0,0,1,0,0],
    "U50": [0,0,0,1,0], 
    "U60": [0,0,0,0,1]
}

# Products
MAX_PRODUCTS = 50  # Sản phẩm mã từ 1->50

# Position encoding
POSITION_MAP = {
    "search": 0,
    "cart": 1,
    "home": 2
}
